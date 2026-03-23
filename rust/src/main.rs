mod bert;

use anyhow::Result;
use bert::BertEmbedder;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

// --- Document types ---

#[derive(Deserialize, Clone)]
pub struct Document {
    pub id: String,
    pub phrase1: String,
    pub phrase2: String,
    #[allow(dead_code)]
    pub metadata: Option<serde_json::Value>,
}

pub struct SearchResult<'a> {
    pub document: &'a Document,
    pub score: f32,
}

fn load_documents(path: &std::path::Path) -> Result<Vec<Document>> {
    let data = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data)?)
}

// --- Vector index ---

struct BruteForceIndex {
    doc_ids: Vec<usize>,
    vectors: Vec<Vec<f32>>,
}

impl BruteForceIndex {
    fn new() -> Self {
        Self { doc_ids: Vec::new(), vectors: Vec::new() }
    }

    fn add(&mut self, doc_id: usize, vector: Vec<f32>) {
        self.doc_ids.push(doc_id);
        self.vectors.push(vector);
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(OrderedFloat, usize)>> =
            std::collections::BinaryHeap::new();

        for (i, vec) in self.vectors.iter().enumerate() {
            let score = dot_product(query, vec);
            let entry = std::cmp::Reverse((OrderedFloat(score), self.doc_ids[i]));
            if heap.len() < top_k {
                heap.push(entry);
            } else if score > heap.peek().unwrap().0 .0 .0 {
                heap.pop();
                heap.push(entry);
            }
        }

        let mut results: Vec<(usize, f32)> = heap.into_iter()
            .map(|std::cmp::Reverse((OrderedFloat(s), id))| (id, s))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    fn size(&self) -> usize {
        self.vectors.len()
    }
}

#[derive(PartialEq, PartialOrd)]
struct OrderedFloat(f32);
impl Eq for OrderedFloat {}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// --- Search engine ---

#[derive(Clone, Copy, Debug)]
enum PhraseStrategy {
    Concatenate,
    Average,
    MaxSim,
}

impl PhraseStrategy {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "concatenate" => Ok(Self::Concatenate),
            "average" => Ok(Self::Average),
            "max_sim" | "maxsim" => Ok(Self::MaxSim),
            _ => anyhow::bail!("Unknown strategy: {}. Use: concatenate, average, max_sim", s),
        }
    }
}

struct DocumentSearchEngine<'a> {
    embedder: &'a BertEmbedder,
    index: BruteForceIndex,
    strategy: PhraseStrategy,
    documents: HashMap<usize, usize>, // internal_id -> doc_index
    next_id: usize,
}

impl<'a> DocumentSearchEngine<'a> {
    fn new(embedder: &'a BertEmbedder, strategy: PhraseStrategy) -> Self {
        Self {
            embedder,
            index: BruteForceIndex::new(),
            strategy,
            documents: HashMap::new(),
            next_id: 0,
        }
    }

    fn index_documents(&mut self, docs: &[Document]) -> Result<()> {
        for (doc_idx, doc) in docs.iter().enumerate() {
            let doc_id = self.next_id;
            self.next_id += 1;
            self.documents.insert(doc_id, doc_idx);

            match self.strategy {
                PhraseStrategy::Concatenate => {
                    let text = format!("{}. {}", doc.phrase1, doc.phrase2);
                    let vec = self.embedder.embed_document(&text)?;
                    self.index.add(doc_id, vec);
                }
                PhraseStrategy::Average => {
                    let v1 = self.embedder.embed_document(&doc.phrase1)?;
                    let v2 = self.embedder.embed_document(&doc.phrase2)?;
                    let mut avg: Vec<f32> = v1.iter().zip(v2.iter())
                        .map(|(a, b)| (a + b) / 2.0)
                        .collect();
                    l2_normalize(&mut avg);
                    self.index.add(doc_id, avg);
                }
                PhraseStrategy::MaxSim => {
                    let v1 = self.embedder.embed_document(&doc.phrase1)?;
                    let v2 = self.embedder.embed_document(&doc.phrase2)?;
                    self.index.add(doc_id, v1);
                    self.index.add(doc_id, v2);
                }
            }
        }
        Ok(())
    }

    fn search<'b>(&self, query: &str, top_k: usize, docs: &'b [Document]) -> Result<Vec<SearchResult<'b>>> {
        let query_vec = self.embedder.embed_query(query)?;

        if matches!(self.strategy, PhraseStrategy::MaxSim) {
            let raw = self.index.search(&query_vec, top_k * 2);
            let mut best: HashMap<usize, f32> = HashMap::new();
            for (id, score) in raw {
                let doc_idx = self.documents[&id];
                let entry = best.entry(doc_idx).or_insert(f32::MIN);
                *entry = entry.max(score);
            }
            let mut results: Vec<_> = best.into_iter().collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            return Ok(results.into_iter()
                .take(top_k)
                .map(|(idx, score)| SearchResult { document: &docs[idx], score })
                .collect());
        }

        let raw = self.index.search(&query_vec, top_k);
        Ok(raw.into_iter()
            .map(|(id, score)| SearchResult {
                document: &docs[self.documents[&id]],
                score,
            })
            .collect())
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

// --- Benchmark ---

const SAMPLE_QUERIES: &[&str] = &[
    "machine learning algorithms",
    "climate change impact",
    "software engineering best practices",
    "healthy cooking recipes",
    "space exploration missions",
    "financial market analysis",
    "artificial intelligence ethics",
    "renewable energy sources",
    "modern web development",
    "quantum computing applications",
];

fn run_benchmark(embedder: &BertEmbedder, docs: &[Document]) -> Result<()> {
    println!("\n=== BENCHMARK ===\n");

    for strategy in [PhraseStrategy::Concatenate, PhraseStrategy::Average, PhraseStrategy::MaxSim] {
        println!("--- Strategy: {:?} ---", strategy);

        let mut engine = DocumentSearchEngine::new(embedder, strategy);

        let t0 = Instant::now();
        engine.index_documents(docs)?;
        let index_ms = t0.elapsed().as_millis();
        println!("  Indexing: {} docs in {} ms ({:.1} ms/doc)",
            docs.len(), index_ms, index_ms as f64 / docs.len() as f64);
        println!("  Index vectors: {}", engine.index.size());

        // Warmup
        for i in 0..3 {
            engine.search(SAMPLE_QUERIES[i % SAMPLE_QUERIES.len()], 5, docs)?;
        }

        // Timed search
        let mut latencies: Vec<u128> = Vec::new();
        for query in SAMPLE_QUERIES {
            let t1 = Instant::now();
            engine.search(query, 5, docs)?;
            latencies.push(t1.elapsed().as_millis());
        }
        latencies.sort();

        println!("  Search latency (ms): min={}, p50={}, p95={}, max={}",
            latencies[0],
            latencies[latencies.len() / 2],
            latencies[(latencies.len() as f64 * 0.95) as usize],
            latencies[latencies.len() - 1]);

        // Sample result
        let sample = engine.search(SAMPLE_QUERIES[0], 3, docs)?;
        println!("  Sample query: \"{}\"", SAMPLE_QUERIES[0]);
        for (i, r) in sample.iter().enumerate() {
            println!("    {}. [{:.4}] {}", i + 1, r.score, r.document.id);
        }
        println!();
    }
    Ok(())
}

// --- CLI ---

struct Args {
    data: PathBuf,
    model_dir: PathBuf,
    query: Option<String>,
    top_k: usize,
    strategy: PhraseStrategy,
    benchmark: bool,
    query_prefix: String,
    doc_prefix: String,
}

fn parse_args() -> Result<Args> {
    let args: Vec<String> = std::env::args().collect();
    let mut result = Args {
        data: PathBuf::from("data/documents.json"),
        model_dir: PathBuf::from("model/berta"),
        query: None,
        top_k: 5,
        strategy: PhraseStrategy::Concatenate,
        benchmark: false,
        query_prefix: "search_query: ".to_string(),
        doc_prefix: "search_document: ".to_string(),
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { i += 1; result.data = PathBuf::from(&args[i]); }
            "--model" | "--model-dir" => { i += 1; result.model_dir = PathBuf::from(&args[i]); }
            "--query" => { i += 1; result.query = Some(args[i].clone()); }
            "--top-k" => { i += 1; result.top_k = args[i].parse()?; }
            "--strategy" => { i += 1; result.strategy = PhraseStrategy::from_str(&args[i])?; }
            "--benchmark" => { result.benchmark = true; }
            "--query-prefix" => { i += 1; result.query_prefix = args[i].clone(); }
            "--doc-prefix" => { i += 1; result.doc_prefix = args[i].clone(); }
            other => anyhow::bail!("Unknown option: {}", other),
        }
        i += 1;
    }

    if !result.benchmark && result.query.is_none() {
        eprintln!("Usage: juq [options]");
        eprintln!("  --data <path>           Path to documents.json");
        eprintln!("  --model <path>          Model directory with .gguf and tokenizer.json");
        eprintln!("  --query <text>          Search query");
        eprintln!("  --top-k <n>             Number of results (default: 5)");
        eprintln!("  --strategy <name>       concatenate|average|max_sim");
        eprintln!("  --benchmark             Run benchmark suite");
        std::process::exit(1);
    }

    Ok(result)
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let t0 = Instant::now();
    println!("Loading model from {:?} ...", args.model_dir);
    let embedder = BertEmbedder::load(&args.model_dir, &args.query_prefix, &args.doc_prefix)?;
    println!("Model loaded in {} ms (dimensions: {})", t0.elapsed().as_millis(), embedder.dimensions());

    let docs = load_documents(&args.data)?;
    println!("Loaded {} documents", docs.len());

    if args.benchmark {
        run_benchmark(&embedder, &docs)?;
    } else {
        let query = args.query.unwrap();
        let mut engine = DocumentSearchEngine::new(&embedder, args.strategy);

        let t1 = Instant::now();
        engine.index_documents(&docs)?;
        println!("Indexed {} documents in {} ms (strategy: {:?})",
            docs.len(), t1.elapsed().as_millis(), args.strategy);

        let t2 = Instant::now();
        let results = engine.search(&query, args.top_k, &docs)?;
        println!("\nQuery: \"{}\" (took {} ms)", query, t2.elapsed().as_millis());
        println!("{}", "─".repeat(60));
        for (i, r) in results.iter().enumerate() {
            println!("{}. [{:.4}] {}", i + 1, r.score, r.document.id);
            println!("   phrase1: {}", r.document.phrase1);
            println!("   phrase2: {}", r.document.phrase2);
        }
    }

    Ok(())
}
