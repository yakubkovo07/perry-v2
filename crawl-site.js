// ══════════════════════════════════════════════
//  crawl-site.js — Peptide Paradise Website Crawler
//
//  This script visits your website, extracts text content
//  from products, pages, and policies, and saves it all
//  to a JSON file (site-content.json).
//
//  This JSON file is then used by index-content.js to build
//  Perry's knowledge base.
//
//  Run with: node scripts/crawl-site.js
// ══════════════════════════════════════════════

import fetch from 'node-fetch';
import * as cheerio from 'cheerio';
import fs from 'fs';

const BASE_URL = 'https://peptideparadiseau.com';

// Pages and collections to crawl.
// Add more URLs here if you have additional pages (FAQ, blog posts, etc.)
const SEED_URLS = [
  `${BASE_URL}/`,
  `${BASE_URL}/collections/all`,
  `${BASE_URL}/collections/products`,
  `${BASE_URL}/pages/shipping`,
  `${BASE_URL}/pages/shipping-policy`,
  `${BASE_URL}/pages/certificate-of-analysis`,
  `${BASE_URL}/pages/faq`,
  `${BASE_URL}/pages/about`,
  `${BASE_URL}/pages/about-us`,
  `${BASE_URL}/pages/contact`,
  `${BASE_URL}/pages/refund-policy`,
  `${BASE_URL}/pages/privacy-policy`,
  `${BASE_URL}/pages/terms-of-service`,
  `${BASE_URL}/policies/refund-policy`,
  `${BASE_URL}/policies/shipping-policy`,
  `${BASE_URL}/policies/privacy-policy`,
  `${BASE_URL}/policies/terms-of-service`,
];

// Sleep helper to avoid hammering the server
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Fetch a single page and extract clean text
async function fetchPage(url) {
  try {
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (PerryBot Knowledge Crawler)' },
    });
    if (!res.ok) {
      console.log(`  Skipped (${res.status}): ${url}`);
      return null;
    }
    const html = await res.text();
    const $ = cheerio.load(html);

    // Remove scripts, styles, nav, footer — we only want content
    $('script, style, nav, footer, header, noscript, svg').remove();

    const title = $('title').text().trim();

    // Try to grab the main content area, fall back to body
    let bodyText = $('main').text() || $('body').text();
    bodyText = bodyText.replace(/\s+/g, ' ').trim();

    return { url, title, content: bodyText };
  } catch (err) {
    console.log(`  Error fetching ${url}: ${err.message}`);
    return null;
  }
}

// Discover all product URLs from the products collection
async function discoverProductUrls() {
  const productUrls = new Set();

  for (let page = 1; page <= 5; page++) {
    const url = `${BASE_URL}/collections/products?page=${page}`;
    console.log(`Discovering products on: ${url}`);
    try {
      const res = await fetch(url, {
        headers: { 'User-Agent': 'Mozilla/5.0 (PerryBot Knowledge Crawler)' },
      });
      if (!res.ok) break;
      const html = await res.text();
      const $ = cheerio.load(html);

      let foundOnPage = 0;
      $('a[href*="/products/"]').each((_, el) => {
        const href = $(el).attr('href');
        if (href) {
          const fullUrl = href.startsWith('http') ? href : `${BASE_URL}${href}`;
          // Strip query params
          const cleanUrl = fullUrl.split('?')[0];
          if (!productUrls.has(cleanUrl)) {
            productUrls.add(cleanUrl);
            foundOnPage++;
          }
        }
      });

      if (foundOnPage === 0) break; // no more pages
      await sleep(500);
    } catch (err) {
      console.log(`  Error on page ${page}: ${err.message}`);
      break;
    }
  }

  return Array.from(productUrls);
}

async function main() {
  console.log('Starting Peptide Paradise website crawl...\n');

  const allUrls = new Set(SEED_URLS);

  // Discover all product pages
  console.log('Discovering product pages...');
  const productUrls = await discoverProductUrls();
  productUrls.forEach((u) => allUrls.add(u));
  console.log(`Found ${productUrls.length} product pages\n`);

  // Crawl every URL
  const results = [];
  let count = 0;
  for (const url of allUrls) {
    count++;
    console.log(`[${count}/${allUrls.size}] Crawling: ${url}`);
    const page = await fetchPage(url);
    if (page && page.content && page.content.length > 50) {
      results.push(page);
    }
    await sleep(400); // be polite to the server
  }

  console.log(`\nCrawled ${results.length} pages successfully.`);

  fs.writeFileSync('site-content.json', JSON.stringify(results, null, 2));
  console.log('Saved to site-content.json');
  console.log('\nNext step: run "node scripts/index-content.js" to build the knowledge base.');
}

main();
