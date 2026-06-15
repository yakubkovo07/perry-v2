// ══════════════════════════════════════════════
//  server.js — Perry Backend (RAG-powered)
//  Peptide Paradise AI Customer Service
//
//  Architecture:
//    1. Customer message arrives
//    2. Check for "personal use" triggers → instant refusal
//    3. Check for "escalation needed" triggers (order-specific) →
//       collect name + order number → direct to email
//    4. Otherwise: search the knowledge base (Pinecone) for
//       relevant website content, then send to Claude with
//       that content as context → stream the response back
//
//  Run with: node server.js
//
//  Required environment variables:
//    ANTHROPIC_API_KEY
//    VOYAGE_API_KEY
//    PINECONE_API_KEY
// ══════════════════════════════════════════════

import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { Pinecone } from '@pinecone-database/pinecone';

const app = express();

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const INDEX_NAME = 'perry-knowledge';

const client = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
const pc = PINECONE_API_KEY ? new Pinecone({ apiKey: PINECONE_API_KEY }) : null;
let pineconeIndex = null;
if (pc) {
  try {
    pineconeIndex = pc.index(INDEX_NAME);
  } catch (e) {
    console.warn('Pinecone index not available yet:', e.message);
  }
}

// ── Middleware ──────────────────────────────────
app.use(
  cors({
    origin: ['https://peptideparadiseau.com', 'http://localhost:3000'],
    methods: ['POST', 'GET'],
    allowedHeaders: ['Content-Type'],
  })
);
app.use(express.json());

const limiter = rateLimit({ windowMs: 60 * 1000, max: 30 });
app.use('/chat', limiter);
app.use('/chat-stream', limiter);

// ── In-memory conversation sessions ────────────
const sessions = new Map();
setInterval(() => {
  const cutoff = Date.now() - 30 * 60 * 1000;
  for (const [id, s] of sessions) {
    if (s.lastActive < cutoff) sessions.delete(id);
  }
}, 30 * 60 * 1000);

// ══════════════════════════════════════════════
//  CLASSIFICATION — decide how to handle a message
// ══════════════════════════════════════════════

// 1. RESEARCH-ONLY triggers — instant refusal, no AI call
const RESEARCH_ONLY_TRIGGERS = [
  'dosage', 'dosing', 'dose', 'how much should i', 'how often should i',
  'how do i inject', 'how to inject', 'injection guide', 'how to administer',
  'administration', 'consume', 'consumption', 'ingest', 'swallow',
  'treatment', 'treat my', 'treat a', 'healing', 'heal my', 'recovery from',
  'recover from', 'weight loss', 'lose weight', 'fat loss', 'burn fat',
  'bodybuilding', 'muscle gain', 'build muscle', 'gain muscle',
  'disease', 'illness', 'condition i have', 'medical outcome',
  'human use', 'for humans', 'inject myself', 'inject it', 'take it',
  'use it on myself', 'should i take', 'what should i take',
  'can i take', 'can i use', 'is it safe to', 'will it help me',
  'for my injury', 'for my pain', 'for my joint', 'for my skin condition',
  'cycle length', 'protocol for', 'stack for', 'how long should i',
  'side effects of taking', 'what happens if i take',
];

// 2. ESCALATION triggers — need customer-specific data
const ESCALATION_TRIGGERS = [
  'track my order', 'tracking number', 'where is my order', "where's my order",
  'order status', 'delivery status', 'shipping update', 'not arrived',
  "hasn't arrived", 'hasnt arrived', 'never arrived', 'not delivered',
  "didn't arrive", 'didnt arrive', 'never received', 'lost my order',
  'lost parcel', 'lost package',
  'missing item', 'wrong item', 'incorrect item', 'not in my order',
  'missing from my order', 'short shipped',
  'damaged order', 'damaged item', 'arrived damaged', 'broken on arrival',
  'return', 'refund', 'exchange', 'send back', 'replacement for my order',
  'payment dispute', 'charged twice', 'double charged', 'overcharged',
  'card declined', 'payment issue with my order', 'billing issue',
  'login issue', "can't log in", 'cant log in', 'forgot my password',
  'account issue', 'my account', 'checkout issue', 'checkout error',
  "can't checkout", 'cant checkout', 'order not going through',
  'discount code not working', 'promo code not working', "code isn't working",
];

const NOISE_WORDS = new Set([
  'my', 'name', 'is', 'i', 'am', 'hi', 'hey', 'hello', 'order', 'number',
  'it', 'the', 'a', 'an', 'and', 'or', 'for', 'with', 'please', 'thanks',
  'thank', 'you', 'can', 'could', 'would', 'should', 'will', 'just',
  'to', 'in', 'of', 'on', 'at', 'from', 'this', 'that', 'these', 'those',
  'yes', 'no', 'ok', 'okay', 're', 'tracking', 'track', 'issue', 'help',
  'support', 'need', 'want', 'have', 'got', 'what', 'when', 'where',
  'who', 'how', 'why', 'about', 'regarding',
]);

function needsResearchDisclaimer(text) {
  const l = text.toLowerCase();
  return RESEARCH_ONLY_TRIGGERS.some((kw) => l.includes(kw));
}

function needsEscalation(text) {
  const l = text.toLowerCase();
  return ESCALATION_TRIGGERS.some((kw) => l.includes(kw));
}

function extractOrderNumber(text) {
  const matches = text.match(/\b(\d{4})\b/g);
  if (!matches) return null;
  for (const m of matches) {
    if (parseInt(m) > 9000) return m;
  }
  return null;
}

function extractName(text) {
  const explicit = text.match(/(?:my name is|name[:\s]+|i(?:'m| am)\s+)([a-zA-Z]+(?:\s[a-zA-Z]+)?)/i);
  if (explicit) return cap(explicit[1].trim());
  const cleaned = text.replace(/\b\d+\b/g, '').replace(/[^a-zA-Z\s]/g, ' ').replace(/\s+/g, ' ').trim();
  const words = cleaned.split(' ').filter((w) => w.length >= 2 && !NOISE_WORDS.has(w.toLowerCase()));
  if (words.length >= 2) return cap(words[0]) + ' ' + cap(words[1]);
  if (words.length === 1) return cap(words[0]);
  return null;
}

function cap(s) {
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

// ══════════════════════════════════════════════
//  RAG — search the knowledge base for relevant content
// ══════════════════════════════════════════════

async function getQueryEmbedding(text) {
  const res = await fetch('https://api.voyageai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${VOYAGE_API_KEY}`,
    },
    body: JSON.stringify({
      input: [text],
      model: 'voyage-3-lite',
      input_type: 'query',
    }),
  });
  if (!res.ok) throw new Error(`Voyage embedding error: ${res.status}`);
  const data = await res.json();
  return data.data[0].embedding;
}

async function searchKnowledgeBase(query, topK = 5) {
  if (!pineconeIndex || !VOYAGE_API_KEY) return [];

  try {
    const embedding = await getQueryEmbedding(query);
    const results = await pineconeIndex.query({
      vector: embedding,
      topK,
      includeMetadata: true,
    });

    return (results.matches || [])
      .filter((m) => m.score > 0.4) // ignore low-relevance results
      .map((m) => ({
        text: m.metadata.text,
        url: m.metadata.url,
        title: m.metadata.title,
        score: m.score,
      }));
  } catch (err) {
    console.error('Knowledge base search error:', err.message);
    return [];
  }
}

// ══════════════════════════════════════════════
//  SYSTEM PROMPT — Perry's core instructions
// ══════════════════════════════════════════════

const BASE_SYSTEM = `You are Perry, the AI customer service assistant for Peptide Paradise — an Australian online retailer of research peptides and bacteriostatic water (peptideparadiseau.com).

You behave like a genuinely helpful human support agent: warm, conversational, and natural — never robotic, never repetitive. You do not use emojis.

══════════════════════
YOUR PRIMARY ROLE
══════════════════════
Answer customer questions directly whenever you can. You will be given relevant content retrieved from the Peptide Paradise website (products, FAQs, shipping policy, certificates, general pages) below — use this as your primary source of truth. If the retrieved content answers the question, answer confidently and naturally, citing the relevant URL if helpful.

Only suggest emailing the team if the question genuinely requires looking up a specific customer's order, payment, shipping, or account information — which you have no access to. Do not redirect to email for general questions about products, policies, shipping rules, or the website, even if the retrieved content doesn't perfectly cover it — use your general knowledge of the business in that case.

══════════════════════
SHIPPING RULES (always correct, use these even if not in retrieved content)
══════════════════════
- Orders placed before 12pm AEST on business days are dispatched the same day
- Orders placed after 12pm AEST, or on weekends/public holidays, are dispatched the next business day
- All orders are sent via AusPost Express
- Free Express Post applies to orders over $200 AUD
- We ship Australia-wide
- Never mention a specific warehouse location or state

══════════════════════
CERTIFICATE OF ANALYSIS
══════════════════════
If asked about Certificates of Analysis, CoA, purity testing, lab testing, third-party testing, or analytical certificates — direct them to:
peptideparadiseau.com/pages/certificate-of-analysis
Then offer to help further if they can't find what they need there.

══════════════════════
PRODUCTS NOT SOLD
══════════════════════
If asked about a peptide we don't stock (e.g. Semaglutide, PT-141, RAD-140, etc.), say something like:
"We don't currently stock that one. You're welcome to browse our full range at peptideparadiseau.com/collections/products."
Do not mention email for this — it's not an escalation case.

══════════════════════
ABSOLUTE COMPLIANCE RULES — NEVER BREAK
══════════════════════
1. NEVER give dosage, injection, administration, consumption, or usage advice of any kind
2. NEVER recommend a peptide for any personal health goal, condition, injury, disease, or outcome
3. NEVER make therapeutic claims. Banned words/phrases: treats, cures, prevents, heals, reverses, repairs, remedy, therapy, clinically proven, guaranteed, doctor recommended, no side effects, medical grade, prescription alternative
4. NEVER associate Retatrutide, Semaglutide, or Tirzepatide with weight loss, fat burning, GLP-1 outcomes, or diabetes
5. NEVER say a peptide is safe for human use, consumption, or injection
6. NEVER mention any specific Australian state or warehouse location
7. Never use emojis

If a question implies personal/human use, dosing, treatment, or medical outcomes (this will often already be filtered before reaching you, but stay vigilant), respond only with:
"Sorry, we cannot assist with that. Our products are strictly supplied for research purposes only."
No further discussion, no alternatives, no speculation.

══════════════════════
TONE
══════════════════════
Sound like a knowledgeable, friendly team member who knows the website inside out. Vary your phrasing naturally — don't use the same sentence structure every time. Keep responses focused and not overly long, but don't be afraid to give a complete, useful answer.`;

// ══════════════════════════════════════════════
//  ROUTES
// ══════════════════════════════════════════════

app.get('/', (req, res) => res.json({ status: 'Perry is online' }));

// ── Non-streaming endpoint (simple request/response) ──
app.post('/chat', async (req, res) => {
  const { message, sessionId, history = [] } = req.body;
  if (!message?.trim() || !sessionId) {
    return res.status(400).json({ error: 'Invalid request' });
  }

  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, { lastActive: Date.now(), awaitingDetails: false, collectedName: null, collectedOrder: null });
  }
  const session = sessions.get(sessionId);
  session.lastActive = Date.now();

  // ── Handle "awaiting details" state (escalation flow) ──
  if (session.awaitingDetails) {
    const foundOrder = extractOrderNumber(message);
    const foundName = extractName(message);
    if (foundOrder) session.collectedOrder = foundOrder;
    if (foundName) session.collectedName = foundName;

    if (!session.collectedOrder && !session.collectedName) {
      return res.json({
        type: 'text',
        reply: "I couldn't find your name or a valid order number in that message. Your order number is a 4-digit number above 9000, found on your confirmation email — could you pop both your name and order number through?",
      });
    }
    if (!session.collectedOrder) {
      return res.json({
        type: 'text',
        reply: `Thanks ${session.collectedName}! I just need your order number too — it's the 4-digit number above 9000 on your confirmation email.`,
      });
    }
    if (!session.collectedName) {
      return res.json({
        type: 'text',
        reply: `Got order #${session.collectedOrder} — could you also let me know your name so our team can find your order?`,
      });
    }

    // Both collected
    const name = session.collectedName;
    const order = session.collectedOrder;
    session.awaitingDetails = false;
    session.collectedName = null;
    session.collectedOrder = null;

    return res.json({
      type: 'success',
      reply: `Thanks ${name}. Please email contact@peptideparadiseau.com with your name (${name}) and order number (#${order}) and our team will assist you as soon as possible.`,
    });
  }

  // ── Step 1: Research-only check ──
  if (needsResearchDisclaimer(message)) {
    return res.json({
      type: 'text',
      reply: 'Sorry, we cannot assist with that. Our products are strictly supplied for research purposes only.',
    });
  }

  // ── Step 2: Escalation check ──
  if (needsEscalation(message)) {
    session.awaitingDetails = true;
    session.collectedName = null;
    session.collectedOrder = null;
    return res.json({
      type: 'escalation',
      reply: "I can help point this in the right direction. Could you let me know your name and order number so our team can look into this for you? Your order number is the 4-digit number above 9000 on your confirmation email.",
    });
  }

  // ── Step 3: RAG — search knowledge base, then answer with Claude ──
  const kbResults = await searchKnowledgeBase(message);
  let contextBlock = '';
  if (kbResults.length > 0) {
    contextBlock = '\n\n══════════════════════\nRELEVANT WEBSITE CONTENT (use this to answer if relevant):\n══════════════════════\n';
    kbResults.forEach((r, i) => {
      contextBlock += `\n[Source ${i + 1}: ${r.title || r.url}]\nURL: ${r.url}\n${r.text}\n`;
    });
  }

  const systemPrompt = BASE_SYSTEM + contextBlock;
  const messages = [...history, { role: 'user', content: message }].slice(-20);

  try {
    const response = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 500,
      system: systemPrompt,
      messages,
    });
    res.json({ type: 'text', reply: response.content[0].text });
  } catch (err) {
    console.error('Anthropic error:', err);
    res.status(500).json({ error: 'Service unavailable' });
  }
});

// ── Streaming endpoint (Server-Sent Events) ──
app.post('/chat-stream', async (req, res) => {
  const { message, sessionId, history = [] } = req.body;
  if (!message?.trim() || !sessionId) {
    return res.status(400).json({ error: 'Invalid request' });
  }

  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, { lastActive: Date.now(), awaitingDetails: false, collectedName: null, collectedOrder: null });
  }
  const session = sessions.get(sessionId);
  session.lastActive = Date.now();

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const send = (type, data) => {
    res.write(`data: ${JSON.stringify({ type, ...data })}\n\n`);
  };

  // ── Awaiting details flow (no streaming needed, instant) ──
  if (session.awaitingDetails) {
    const foundOrder = extractOrderNumber(message);
    const foundName = extractName(message);
    if (foundOrder) session.collectedOrder = foundOrder;
    if (foundName) session.collectedName = foundName;

    let reply, type = 'text';
    if (!session.collectedOrder && !session.collectedName) {
      reply = "I couldn't find your name or a valid order number in that message. Your order number is a 4-digit number above 9000, found on your confirmation email — could you pop both your name and order number through?";
    } else if (!session.collectedOrder) {
      reply = `Thanks ${session.collectedName}! I just need your order number too — it's the 4-digit number above 9000 on your confirmation email.`;
    } else if (!session.collectedName) {
      reply = `Got order #${session.collectedOrder} — could you also let me know your name so our team can find your order?`;
    } else {
      const name = session.collectedName;
      const order = session.collectedOrder;
      session.awaitingDetails = false;
      session.collectedName = null;
      session.collectedOrder = null;
      type = 'success';
      reply = `Thanks ${name}. Please email contact@peptideparadiseau.com with your name (${name}) and order number (#${order}) and our team will assist you as soon as possible.`;
    }

    send('start', { responseType: type });
    send('delta', { text: reply });
    send('done', {});
    return res.end();
  }

  // ── Research-only check ──
  if (needsResearchDisclaimer(message)) {
    send('start', { responseType: 'text' });
    send('delta', { text: 'Sorry, we cannot assist with that. Our products are strictly supplied for research purposes only.' });
    send('done', {});
    return res.end();
  }

  // ── Escalation check ──
  if (needsEscalation(message)) {
    session.awaitingDetails = true;
    session.collectedName = null;
    session.collectedOrder = null;
    send('start', { responseType: 'escalation' });
    send('delta', { text: "I can help point this in the right direction. Could you let me know your name and order number so our team can look into this for you? Your order number is the 4-digit number above 9000 on your confirmation email." });
    send('done', {});
    return res.end();
  }

  // ── RAG + streaming AI response ──
  const kbResults = await searchKnowledgeBase(message);
  let contextBlock = '';
  let sources = [];
  if (kbResults.length > 0) {
    contextBlock = '\n\n══════════════════════\nRELEVANT WEBSITE CONTENT (use this to answer if relevant):\n══════════════════════\n';
    kbResults.forEach((r, i) => {
      contextBlock += `\n[Source ${i + 1}: ${r.title || r.url}]\nURL: ${r.url}\n${r.text}\n`;
      sources.push({ title: r.title, url: r.url });
    });
  }

  const systemPrompt = BASE_SYSTEM + contextBlock;
  const messages = [...history, { role: 'user', content: message }].slice(-20);

  try {
    send('start', { responseType: 'text', sources });

    const stream = await client.messages.stream({
      model: 'claude-sonnet-4-6',
      max_tokens: 500,
      system: systemPrompt,
      messages,
    });

    let fullText = '';
    for await (const event of stream) {
      if (event.type === 'content_block_delta' && event.delta?.text) {
        fullText += event.delta.text;
        send('delta', { text: event.delta.text });
      }
    }

    send('done', { fullText });
  } catch (err) {
    console.error('Streaming error:', err);
    send('error', { message: 'Something went wrong. Please try again.' });
  }

  res.end();
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Perry backend (RAG) running on port ${PORT}`));
