// ══════════════════════════════════════════════
//  index-content.js — Build Perry's Knowledge Base
//
//  This script takes site-content.json (from crawl-site.js),
//  splits it into chunks, generates embeddings using Voyage AI,
//  and uploads everything to Pinecone.
//
//  This is what allows Perry to "search" your website content
//  before answering — RAG (Retrieval Augmented Generation).
//
//  Run with: node scripts/index-content.js
//
//  Requires these environment variables:
//    VOYAGE_API_KEY   - from https://dash.voyageai.com
//    PINECONE_API_KEY - from https://app.pinecone.io
// ══════════════════════════════════════════════

import fs from 'fs';
import { Pinecone } from '@pinecone-database/pinecone';

const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const INDEX_NAME = 'perry-knowledge';

if (!VOYAGE_API_KEY || !PINECONE_API_KEY) {
  console.error('ERROR: Missing VOYAGE_API_KEY or PINECONE_API_KEY environment variables.');
  console.error('Set them before running this script, e.g.:');
  console.error('  export VOYAGE_API_KEY=your_key');
  console.error('  export PINECONE_API_KEY=your_key');
  process.exit(1);
}

// ── Split text into chunks of roughly 'maxLength' characters ──
function chunkText(text, maxLength = 1200, overlap = 150) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + maxLength, text.length);

    // Try to break at a sentence boundary
    if (end < text.length) {
      const lastPeriod = text.lastIndexOf('. ', end);
      if (lastPeriod > start + maxLength * 0.5) {
        end = lastPeriod + 1;
      }
    }

    chunks.push(text.slice(start, end).trim());
    start = end - overlap;
    if (start < 0) start = end;
    if (end >= text.length) break;
  }
  return chunks.filter((c) => c.length > 30);
}

// ── Generate embeddings using Voyage AI ──
async function getEmbeddings(texts) {
  const res = await fetch('https://api.voyageai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${VOYAGE_API_KEY}`,
    },
    body: JSON.stringify({
      input: texts,
      model: 'voyage-3-lite', // cheap, fast, great for this use case
      input_type: 'document',
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Voyage API error: ${res.status} ${err}`);
  }

  const data = await res.json();
  return data.data.map((d) => d.embedding);
}

async function main() {
  console.log('Loading site-content.json...');
  if (!fs.existsSync('site-content.json')) {
    console.error('ERROR: site-content.json not found.');
    console.error('Run "node scripts/crawl-site.js" first.');
    process.exit(1);
  }

  const pages = JSON.parse(fs.readFileSync('site-content.json', 'utf-8'));
  console.log(`Loaded ${pages.length} pages.\n`);

  // Build all chunks across all pages
  const allChunks = [];
  for (const page of pages) {
    const chunks = chunkText(page.content);
    chunks.forEach((chunk, i) => {
      allChunks.push({
        id: `${page.url}#${i}`,
        text: chunk,
        url: page.url,
        title: page.title,
      });
    });
  }
  console.log(`Created ${allChunks.length} text chunks.\n`);

  // Connect to Pinecone
  console.log('Connecting to Pinecone...');
  const pc = new Pinecone({ apiKey: PINECONE_API_KEY });

  // Check if index exists, create if not
  const existingIndexes = await pc.listIndexes();
  const indexExists = existingIndexes.indexes?.some((idx) => idx.name === INDEX_NAME);

  if (!indexExists) {
    console.log(`Creating Pinecone index "${INDEX_NAME}"...`);
    await pc.createIndex({
      name: INDEX_NAME,
      dimension: 512, // voyage-3-lite dimension
      metric: 'cosine',
      spec: {
        serverless: {
          cloud: 'aws',
          region: 'us-east-1',
        },
      },
    });
    // Wait for index to be ready
    console.log('Waiting for index to initialize...');
    await new Promise((r) => setTimeout(r, 10000));
  } else {
    console.log(`Index "${INDEX_NAME}" already exists.`);
  }

  const index = pc.index(INDEX_NAME);

  // Process in batches of 50 (Voyage API limit per request)
  const BATCH_SIZE = 50;
  for (let i = 0; i < allChunks.length; i += BATCH_SIZE) {
    const batch = allChunks.slice(i, i + BATCH_SIZE);
    console.log(`Embedding batch ${Math.floor(i / BATCH_SIZE) + 1} / ${Math.ceil(allChunks.length / BATCH_SIZE)}...`);

    const texts = batch.map((c) => c.text);
    const embeddings = await getEmbeddings(texts);

    const vectors = batch.map((chunk, j) => ({
      id: chunk.id,
      values: embeddings[j],
      metadata: {
        text: chunk.text,
        url: chunk.url,
        title: chunk.title || '',
      },
    }));

    await index.upsert(vectors);
    console.log(`  Uploaded ${vectors.length} vectors.`);

    // Small delay to respect rate limits
    await new Promise((r) => setTimeout(r, 500));
  }

  console.log('\nDone! Perry\'s knowledge base is ready.');
  console.log(`Total chunks indexed: ${allChunks.length}`);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
