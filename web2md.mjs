import { Readability } from '@mozilla/readability';
import jsdom from 'jsdom';
import fetch from 'node-fetch';
import TurndownService from 'turndown';

async function getMainContent(url) {
  const response = await fetch(url);
  const body = await response.text();
  const doc = new jsdom.JSDOM(body).window.document;
  const reader = new Readability(doc);
  const article = reader.parse();
  // console.log(article.content);

  let turndownService = new TurndownService();
  let markdown = turndownService.turndown(article.content);

  console.log(markdown);
}

const url = process.argv[2];
if (url) {
  getMainContent(url);
} else {
  console.error('Please provide a URL as an argument');
}
