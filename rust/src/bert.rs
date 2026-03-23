use anyhow::{bail, Result};
use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::LayerNorm;
use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;
use tokenizers::Tokenizer;

struct QLinear {
    weight: Tensor,
    bias: Tensor,
}

impl QLinear {
    fn new(weight: QTensor, bias: Tensor, device: &Device) -> Result<Self> {
        // Pre-dequantize and pre-transpose for fast inference
        let weight = weight.dequantize(device)?.t()?.contiguous()?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        Ok(out.broadcast_add(&self.bias)?)
    }
}

struct BertEmbeddings {
    word_embeddings: Tensor,
    position_embeddings: Tensor,
    token_type_embeddings: Tensor,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn forward(&self, input_ids: &[u32], token_type_ids: &[u32], device: &Device) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let ids = Tensor::new(input_ids, device)?;
        let type_ids = Tensor::new(token_type_ids, device)?;
        let position_ids = Tensor::arange(0u32, seq_len as u32, device)?;

        let word_emb = self.word_embeddings.index_select(&ids, 0)?;
        let pos_emb = self.position_embeddings.index_select(&position_ids, 0)?;
        let type_emb = self.token_type_embeddings.index_select(&type_ids, 0)?;

        let embeddings = (word_emb + pos_emb + type_emb)?;
        Ok(self.layer_norm.forward(&embeddings)?)
    }
}

struct BertLayer {
    query: QLinear,
    key: QLinear,
    value: QLinear,
    attn_output: QLinear,
    attn_output_norm: LayerNorm,
    ffn_up: QLinear,
    ffn_down: QLinear,
    layer_output_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl BertLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(0)?;

        // Self-attention
        let q = self.query.forward(x)?
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;
        let k = self.key.forward(x)?
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;
        let v = self.value.forward(x)?
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;

        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let scores = q.contiguous()?.matmul(&k_t)?;
        let scores = (scores / scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn = attn_weights.matmul(&v.contiguous()?)?;

        let attn = attn.transpose(0, 1)?
            .contiguous()?
            .reshape((seq_len, self.num_heads * self.head_dim))?;

        let attn = self.attn_output.forward(&attn)?;
        let x = self.attn_output_norm.forward(&(x + attn)?)?;

        // Feed-forward
        let ffn = self.ffn_up.forward(&x)?;
        let ffn = ffn.gelu_erf()?;
        let ffn = self.ffn_down.forward(&ffn)?;
        let x = self.layer_output_norm.forward(&(x + ffn)?)?;

        Ok(x)
    }
}

pub struct BertModel {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
    device: Device,
}

impl BertModel {
    pub fn load(gguf_path: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let mut reader = BufReader::new(std::fs::File::open(gguf_path)?);
        let content = gguf_file::Content::read(&mut reader)?;

        let md = &content.metadata;
        let num_layers = get_metadata_u32(md, "bert.block_count")? as usize;
        let hidden_size = get_metadata_u32(md, "bert.embedding_length")? as usize;
        let num_heads = get_metadata_u32(md, "bert.attention.head_count")? as usize;
        let layer_norm_eps = get_metadata_f32(md, "bert.attention.layer_norm_epsilon")? as f64;
        let head_dim = hidden_size / num_heads;

        // Helper to load tensors
        let mut tensors: HashMap<String, QTensor> = HashMap::new();
        for tensor_info in &content.tensor_infos {
            let name = tensor_info.0.clone();
            let qt = content.tensor(&mut reader, &name, &device)?;
            tensors.insert(name, qt);
        }

        fn take_f32(tensors: &mut HashMap<String, QTensor>, name: &str, device: &Device) -> Result<Tensor> {
            Ok(tensors.remove(name)
                .ok_or_else(|| anyhow::anyhow!("tensor not found: {}", name))?
                .dequantize(device)?)
        }

        fn take_qt(tensors: &mut HashMap<String, QTensor>, name: &str) -> Result<QTensor> {
            tensors.remove(name)
                .ok_or_else(|| anyhow::anyhow!("tensor not found: {}", name))
        }

        // Embeddings
        let word_emb = take_f32(&mut tensors, "token_embd.weight", &device)?;
        let pos_emb = take_f32(&mut tensors, "position_embd.weight", &device)?;
        let type_emb = take_f32(&mut tensors, "token_types.weight", &device)?;
        let emb_norm = LayerNorm::new(
            take_f32(&mut tensors, "token_embd_norm.weight", &device)?,
            take_f32(&mut tensors, "token_embd_norm.bias", &device)?,
            layer_norm_eps,
        );

        let embeddings = BertEmbeddings {
            word_embeddings: word_emb,
            position_embeddings: pos_emb,
            token_type_embeddings: type_emb,
            layer_norm: emb_norm,
        };

        // Transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("blk.{}", i);
            let layer = BertLayer {
                query: QLinear::new(
                    take_qt(&mut tensors, &format!("{prefix}.attn_q.weight"))?,
                    take_f32(&mut tensors, &format!("{prefix}.attn_q.bias"), &device)?,
                    &device,
                )?,
                key: QLinear::new(
                    take_qt(&mut tensors, &format!("{prefix}.attn_k.weight"))?,
                    take_f32(&mut tensors, &format!("{prefix}.attn_k.bias"), &device)?,
                    &device,
                )?,
                value: QLinear::new(
                    take_qt(&mut tensors, &format!("{prefix}.attn_v.weight"))?,
                    take_f32(&mut tensors, &format!("{prefix}.attn_v.bias"), &device)?,
                    &device,
                )?,
                attn_output: QLinear::new(
                    take_qt(&mut tensors, &format!("{prefix}.attn_output.weight"))?,
                    take_f32(&mut tensors, &format!("{prefix}.attn_output.bias"), &device)?,
                    &device,
                )?,
                attn_output_norm: LayerNorm::new(
                    take_f32(&mut tensors, &format!("{prefix}.attn_output_norm.weight"), &device)?,
                    take_f32(&mut tensors, &format!("{prefix}.attn_output_norm.bias"), &device)?,
                    layer_norm_eps,
                ),
                ffn_up: QLinear::new(
                    take_qt(&mut tensors, &format!("{prefix}.ffn_up.weight"))?,
                    take_f32(&mut tensors, &format!("{prefix}.ffn_up.bias"), &device)?,
                    &device,
                )?,
                ffn_down: QLinear::new(
                    take_qt(&mut tensors, &format!("{prefix}.ffn_down.weight"))?,
                    take_f32(&mut tensors, &format!("{prefix}.ffn_down.bias"), &device)?,
                    &device,
                )?,
                layer_output_norm: LayerNorm::new(
                    take_f32(&mut tensors, &format!("{prefix}.layer_output_norm.weight"), &device)?,
                    take_f32(&mut tensors, &format!("{prefix}.layer_output_norm.bias"), &device)?,
                    layer_norm_eps,
                ),
                num_heads,
                head_dim,
            };
            layers.push(layer);
        }

        Ok(Self { embeddings, layers, device })
    }

    pub fn forward(&self, input_ids: &[u32], token_type_ids: &[u32]) -> Result<Tensor> {
        let mut x = self.embeddings.forward(input_ids, token_type_ids, &self.device)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    query_prefix: String,
    doc_prefix: String,
    dims: usize,
}

impl BertEmbedder {
    pub fn load(model_dir: &Path, query_prefix: &str, doc_prefix: &str) -> Result<Self> {
        let gguf_files: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "gguf"))
            .collect();

        if gguf_files.is_empty() {
            bail!("No .gguf file found in {:?}", model_dir);
        }
        let gguf_path = gguf_files[0].path();

        let model = BertModel::load(&gguf_path)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Probe dimensions
        let dims = {
            let encoding = tokenizer.encode("probe", true)
                .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;
            let ids = encoding.get_ids().to_vec();
            let type_ids = encoding.get_type_ids().to_vec();
            let output = model.forward(&ids, &type_ids)?;
            output.dim(1)?
        };

        Ok(Self {
            model,
            tokenizer,
            query_prefix: query_prefix.to_string(),
            doc_prefix: doc_prefix.to_string(),
            dims,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dims
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {}", e))?;

        let ids = encoding.get_ids().to_vec();
        let type_ids = encoding.get_type_ids().to_vec();
        let mask = encoding.get_attention_mask().to_vec();

        let hidden = self.model.forward(&ids, &type_ids)?;

        // Mean pooling
        let pooled = mean_pool(&hidden, &mask)?;

        // L2 normalize
        let norm = pooled.sqr()?.sum_all()?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?;

        Ok(normalized.to_vec1::<f32>()?)
    }

    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let text = format!("{}{}", self.query_prefix, query);
        self.embed(&text)
    }

    pub fn embed_document(&self, doc: &str) -> Result<Vec<f32>> {
        let text = format!("{}{}", self.doc_prefix, doc);
        self.embed(&text)
    }

    pub fn query_prefix(&self) -> &str {
        &self.query_prefix
    }

    pub fn doc_prefix(&self) -> &str {
        &self.doc_prefix
    }
}

fn mean_pool(hidden: &Tensor, attention_mask: &[u32]) -> Result<Tensor> {
    let device = hidden.device();

    let mask = Tensor::new(attention_mask, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?; // [seq_len, 1]

    let masked = hidden.broadcast_mul(&mask)?;
    let sum = masked.sum(0)?; // [hidden_size]
    let count = mask.sum_all()?;
    Ok(sum.broadcast_div(&count)?)
}

fn get_metadata_u32(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<u32> {
    match md.get(key) {
        Some(gguf_file::Value::U32(v)) => Ok(*v),
        Some(gguf_file::Value::U16(v)) => Ok(*v as u32),
        Some(gguf_file::Value::U8(v)) => Ok(*v as u32),
        Some(v) => bail!("metadata {key} has unexpected type: {:?}", v),
        None => bail!("metadata {key} not found"),
    }
}

fn get_metadata_f32(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<f32> {
    match md.get(key) {
        Some(gguf_file::Value::F32(v)) => Ok(*v),
        Some(gguf_file::Value::F64(v)) => Ok(*v as f32),
        Some(v) => bail!("metadata {key} has unexpected type: {:?}", v),
        None => bail!("metadata {key} not found"),
    }
}
