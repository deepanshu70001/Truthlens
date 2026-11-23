const { chromium } = require('d:/Projects/Truthlens/frontend/node_modules/@playwright/test');
const path = require('path');
const fs = require('fs');

(async () => {
  console.log('=== WORKFLOW B: Live E2E with Synthetic Report Enrichment DISABLED (Baseline) ===');
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
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

  fs.mkdirSync('docs/screenshots', { recursive: true });

  console.log('1. Navigating to http://127.0.0.1:5173/...');
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
  await page.screenshot({ path: 'docs/screenshots/workflow_b_01_landing.png' });

  console.log('2. Uploading test fixture data/test_fixtures/e2e_live_test.jpg...');
  const fixturePath = path.resolve('data/test_fixtures/e2e_live_test.jpg');
  await page.locator('input[type="file"]').setInputFiles(fixturePath);

  await page.waitForTimeout(500);
  await page.screenshot({ path: 'docs/screenshots/workflow_b_02_submitted.png' });

  console.log('3. Clicking Analyze button...');
  const analyzeBtn = page.getByRole('button', { name: /Analyze/i });
  await analyzeBtn.click();

  console.log('4. Waiting for analysis submission response...');
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

  console.log('5. Waiting for worker to complete pipeline and report to render in browser...');
  try {
    await page.waitForSelector('text=Analysis Summary', { timeout: 120_000 });
    console.log('SUCCESS: Analysis Summary is visible in UI!');
  } catch (err) {
    console.error('Timed out waiting for Analysis Summary:', err.message);
    await page.screenshot({ path: 'docs/screenshots/workflow_b_error_timeout.png', fullPage: true });
    await browser.close();
    process.exit(1);
  }

  await page.waitForTimeout(2000);

  console.log('6. Verifying baseline non-synthetic status in the DOM...');
  const baselineCheck = await page.evaluate(() => {
    const bodyText = document.body.innerText;
    return {
      hasIllustrativeHeader: bodyText.includes('Illustrative Content Context (Synthetic Demonstration)'),
      hasSyntheticDisclaimer: bodyText.includes('SYNTHETIC DEMONSTRATION DATA'),
      hasViralityDemo: bodyText.includes('Content Virality Indicators (Synthetic Demo)'),
      hasViralityHeuristic: bodyText.includes('Content Spread Indicators (Heuristic)'),
      hasTrajectorySection: bodyText.includes('Content Spread Trajectory (Simulated Demonstration)'),
      hasRegionalUnavailable: bodyText.includes('Regional scores and rankings are unavailable'),
      hasRegionalSimulation: bodyText.includes('India — Regional Exposure Simulation (Synthetic Demo)'),
      hasDownloadButton: Array.from(document.querySelectorAll('button')).some(b => b.innerText.includes('Download Report')),
    };
  });

  console.log('Baseline Check Results:', baselineCheck);

  if (baselineCheck.hasIllustrativeHeader || baselineCheck.hasTrajectorySection || baselineCheck.hasRegionalSimulation) {
    console.error('ASSERTION FAILED: Baseline report contains unexpected synthetic elements!');
  }

  console.log('7. Capturing Workflow B report screenshots...');
  await page.screenshot({ path: 'docs/screenshots/workflow_b_03_report_baseline.png' });

  // Scroll to spread section
  const spreadEl = page.locator('text=Content Spread Indicators (Heuristic)');
  if (await spreadEl.count() > 0) {
    await spreadEl.first().scrollIntoViewIfNeeded();
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'docs/screenshots/workflow_b_04_spread_unavailable.png' });
  }

  // Test Download Report button
  console.log('8. Testing Download Report button in Workflow B...');
  const downloadPromise = page.waitForEvent('download', { timeout: 10_000 }).catch(() => null);
  const downloadBtn = page.getByRole('button', { name: /Download Report/i });
  if (await downloadBtn.count() > 0) {
    await downloadBtn.first().click();
    const download = await downloadPromise;
    if (download) {
      const downloadPath = path.resolve('docs/screenshots/downloaded_report_workflow_b.json');
      await download.saveAs(downloadPath);
      console.log('Downloaded report successfully saved to:', downloadPath);
    }
  }

  await page.screenshot({ path: 'docs/screenshots/workflow_b_05_report_full.png', fullPage: true });

  const resultSummary = {
    workflow: 'Workflow B (Synthetic Report Enrichment DISABLED - Baseline)',
    analysisId,
    baselineCheck,
    networkRequestsCount: networkLog.length,
    networkLog: networkLog.map(n => ({ url: n.url, method: n.method, status: n.status })),
    consoleErrors: consoleMessages.filter(m => m.type === 'error'),
  };

  fs.writeFileSync('docs/screenshots/workflow_b_result.json', JSON.stringify(resultSummary, null, 2));
  console.log('Saved docs/screenshots/workflow_b_result.json');
  console.log('Workflow B finished successfully!');

  await browser.close();
})();
