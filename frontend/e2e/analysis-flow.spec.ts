import { expect, test } from '@playwright/test'

const analysisId = '6a522bf5-32b1-4e4a-8b6d-901f77d270ad'

test('authenticate, submit an image job, poll durable status, and view its report', async ({ page }) => {
  await page.addInitScript(() => {
    const original = window.setInterval.bind(window)
    window.setInterval = ((handler: TimerHandler, timeout?: number, ...args: unknown[]) =>
      original(handler, Math.min(Number(timeout ?? 0), 25), ...args)) as typeof window.setInterval
  })

  await page.route('**/api/v1/auth/mode', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ auth_required: true }),
  }))
  await page.route('**/api/v1/auth/token', async route => {
    expect(route.request().method()).toBe('POST')
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ access_token: 'mock-signed-token', token_type: 'bearer', expires_in: 3600 }),
    })
  })
  await page.route('**/api/v1/analyze', async route => {
    expect(route.request().headers().authorization).toBe('Bearer mock-signed-token')
    expect(route.request().headers()['idempotency-key']).toBeTruthy()
    await route.fulfill({
      status: 201,
      contentType: 'application/json',
      body: JSON.stringify({ analysis_id: analysisId, job_id: analysisId, status: 'queued' }),
    })
  })
  await page.route(`**/api/v1/analysis/${analysisId}`, route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      analysis_id: analysisId,
      status: 'partial',
      job: { job_id: analysisId, state: 'completed', attempt_count: 1, max_attempts: 3 },
      progress: { completed_steps: ['ocr'], stage_statuses: { ocr: 'completed', provenance: 'unavailable' } },
    }),
  }))
  await page.route(`**/api/v1/analysis/${analysisId}/report`, route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ analysis_id: analysisId, status: 'partial', stages: {}, summary: {} }),
  }))
  await page.route(`**/api/v1/analysis/${analysisId}/image`, route => route.fulfill({
    status: 200,
    contentType: 'image/jpeg',
    body: Buffer.from(
      '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAb/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdAB//2Q==',
      'base64',
    ),
  }))

  await page.goto('/')
  await page.getByLabel('Username').fill('test-user')
  await page.getByLabel('Password').fill('test-password')
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page.getByText('Drop an image here or click to select')).toBeVisible()

  await page.locator('input[type="file"]').setInputFiles({
    name: 'upload.jpg',
    mimeType: 'image/jpeg',
    buffer: Buffer.from('mock image bytes'),
  })
  await page.getByRole('button', { name: 'Analyze' }).click()
  await expect(page.getByText('Analysis Summary')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByText(/Analysis status: partial/)).toBeVisible()
})
