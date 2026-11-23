const { chromium } = require('d:/Projects/Truthlens/frontend/node_modules/@playwright/test');
const path = require('path');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  const networkLog = [];
  const consoleMessages = [];

  page.on('console', msg => {
    consoleMessages.push({ type: msg.type(), text: msg.text() });
  });

  page.on('response', async resp => {
    const url = resp.url();
    if (url.includes('/api/v1/')) {
      let bodyText = '';
      try {
        bodyText = await resp.text();
      } catch (e) {
        bodyText = '<unreadable>';
      }
      networkLog.push({
        url,
        status: resp.status(),
        method: resp.request().method(),
        body: bodyText.length > 2000 ? bodyText.slice(0, 2000) + '...[truncated]' : bodyText,
      });
    }
  });

  console.log('1. Navigating to http://127.0.0.1:5173/...');
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
  await page.screenshot({ path: 'docs/screenshots/01_landing_page.png' });

  console.log('2. Uploading test fixture data/test_fixtures/e2e_live_test.jpg...');
  const fixturePath = path.resolve('data/test_fixtures/e2e_live_test.jpg');
  await page.locator('input[type="file"]').setInputFiles(fixturePath);

  await page.waitForTimeout(500);
  await page.screenshot({ path: 'docs/screenshots/02_image_selected.png' });

  console.log('3. Clicking Analyze button...');
  const analyzeBtn = page.getByRole('button', { name: /Analyze/i });
  await analyzeBtn.click();

  console.log('4. Waiting for analysis submission response...');
  // Wait for submission response
  let analysisId = null;
  for (let i = 0; i < 20; i++) {
    const submitReq = networkLog.find(r => r.url.includes('/api/v1/analyze') && r.method === 'POST');
    if (submitReq && submitReq.status === 201) {
      try {
        const parsed = JSON.parse(submitReq.body);
        analysisId = parsed.analysis_id;
        console.log('Received analysisId:', analysisId, 'Status:', parsed.status);
        break;
      } catch (e) {}
    }
    await page.waitForTimeout(500);
  }

  await page.screenshot({ path: 'docs/screenshots/03_analysis_processing.png' });

  console.log('5. Waiting for worker to complete pipeline and report to render in browser...');
  // Wait for report summary to become visible (timeout 120s for full pipeline)
  try {
    await page.waitForSelector('text=Analysis Summary', { timeout: 120_000 });
    console.log('SUCCESS: Analysis Summary is visible in UI!');
  } catch (err) {
    console.error('Timed out waiting for Analysis Summary:', err.message);
    await page.screenshot({ path: 'docs/screenshots/error_timeout.png', fullPage: true });
    await browser.close();
    process.exit(1);
  }

  // Allow animations and render to settle
  await page.waitForTimeout(2000);

  console.log('6. Capturing final report screenshots...');
  await page.screenshot({ path: 'docs/screenshots/04_report_summary.png' });
  await page.screenshot({ path: 'docs/screenshots/05_report_full.png', fullPage: true });

  console.log('7. Extracting rendered UI details from the DOM...');
  const reportData = await page.evaluate(() => {
    const textOf = sel => document.querySelector(sel)?.innerText?.trim() || null;
    const allTextOf = sel => Array.from(document.querySelectorAll(sel)).map(el => el.innerText.trim());

    return {
      statusBadge: document.querySelector('[class*="rounded-full"]')?.innerText || null,
      summaryCards: allTextOf('.grid > div'),
      bodyText: document.body.innerText.slice(0, 3000),
      headings: allTextOf('h1, h2, h3, h4'),
      hasImage: Boolean(document.querySelector('img[alt="Uploaded"]')),
      tabs: allTextOf('button[role="tab"], nav button'),
    };
  });

  const resultSummary = {
    analysisId,
    uiReportData: reportData,
    networkRequestsCount: networkLog.length,
    networkLog: networkLog.map(n => ({ url: n.url, method: n.method, status: n.status })),
    consoleErrors: consoleMessages.filter(m => m.type === 'error'),
  };

  fs.writeFileSync('docs/screenshots/live_e2e_result.json', JSON.stringify(resultSummary, null, 2));
  console.log('Saved docs/screenshots/live_e2e_result.json');
  console.log('Live E2E test workflow finished successfully!');

  await browser.close();
})();
