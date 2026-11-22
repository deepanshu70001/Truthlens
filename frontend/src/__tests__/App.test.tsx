import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'

const api = vi.hoisted(() => ({
  getAuthMode: vi.fn(),
  getAccessToken: vi.fn(),
  getAnalysisStatus: vi.fn(),
  getAnalysisReport: vi.fn(),
  getAnalysisImageUrl: vi.fn(),
  getHeatmapUrl: vi.fn(),
  getProtectedAsset: vi.fn(),
  login: vi.fn(),
  logout: vi.fn(),
  uploadImage: vi.fn(),
}))

vi.mock('../services/api', () => api)

const analysisId = '25ab31dc-0346-4235-a2c9-2c56117bb5ee'

beforeEach(() => {
  api.getAuthMode.mockResolvedValue({ auth_required: true })
  api.getAccessToken.mockReturnValue(null)
  api.login.mockResolvedValue(undefined)
  api.logout.mockImplementation(() => undefined)
  api.uploadImage.mockResolvedValue({ analysis_id: analysisId, job_id: analysisId, status: 'queued' })
  api.getAnalysisStatus.mockResolvedValue({
    analysis_id: analysisId,
    status: 'partial',
    progress: { completed_steps: ['ocr'], stage_statuses: { ocr: 'completed', provenance: 'unavailable' } },
  })
  api.getAnalysisReport.mockResolvedValue({ analysis_id: analysisId, status: 'partial', stages: {}, summary: {} })
  api.getProtectedAsset.mockResolvedValue(new Blob(['image']))
  vi.stubGlobal('crypto', { randomUUID: () => 'e10-test-idempotency-key' })
  window.history.replaceState({}, '', '/')
})

describe('authenticated analysis flow', () => {
  it('shows sign-in first, rejects invalid files, submits, polls, and renders a partial report', async () => {
    render(<App />)
    expect(await screen.findByText('Sign in to your workspace')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'alice' } })
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'password' } })
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }))
    expect(await screen.findByText('Drop an image here or click to select')).toBeInTheDocument()

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [new File(['text'], 'not-image.txt', { type: 'text/plain' })] } })
    expect(await screen.findByText('Please select an image file.')).toBeInTheDocument()

    fireEvent.change(input, { target: { files: [new File(['jpeg-bytes'], 'photo.jpg', { type: 'image/jpeg' })] } })
    fireEvent.click(screen.getByRole('button', { name: 'Analyze' }))
    await waitFor(() => expect(api.uploadImage).toHaveBeenCalledOnce())
    expect(api.uploadImage.mock.calls[0][1]).toBe('e10-test-idempotency-key')
    expect(await screen.findByText('Analysis Progress')).toBeInTheDocument()
    expect(sessionStorage.getItem('truthlens.analysisId')).toBe(analysisId)

    await waitFor(() => expect(api.getAnalysisReport).toHaveBeenCalledOnce(), { timeout: 8_000 })
    expect(await screen.findByText(/Analysis status: partial/, {}, { timeout: 2_000 })).toBeInTheDocument()
    expect(api.getAnalysisStatus).toHaveBeenCalledWith(analysisId)
    expect(api.getAnalysisReport).toHaveBeenCalledWith(analysisId)
  })

  it('restores a completed report after a page reload', async () => {
    window.history.replaceState({}, '', `/?analysis=${analysisId}`)
    api.getAuthMode.mockResolvedValue({ auth_required: false })
    api.getAnalysisStatus.mockResolvedValue({ analysis_id: analysisId, status: 'completed' })
    api.getAnalysisReport.mockResolvedValue({
      analysis_id: analysisId,
      status: 'completed',
      stages: {},
      summary: {},
      assessment: { label: 'inconclusive', overall_score: null, confidence: null },
      fact_checks: { status: 'no_results', claims: [] },
      provenance: { status: 'unavailable' },
    })

    render(<App />)
    expect(await screen.findByText('Analysis Summary')).toBeInTheDocument()
    expect(screen.getByText('OCR result unavailable: this report contains no OCR result.')).toBeInTheDocument()
    expect(screen.getByText('No conclusive review was available for some or all eligible claims.')).toBeInTheDocument()
    expect(screen.getByText('Web detection is unavailable because provider credentials or service support are missing.')).toBeInTheDocument()
    expect(screen.getByText('Detailed evidence dispositions are unavailable in this historical report. The legacy assessment remains readable.')).toBeInTheDocument()
    expect(api.getAnalysisStatus).toHaveBeenCalledWith(analysisId)
    expect(api.getAnalysisReport).toHaveBeenCalledWith(analysisId)
  })

  it('returns to sign-in when the session expires', async () => {
    api.getAuthMode.mockResolvedValue({ auth_required: true })
    api.getAccessToken.mockReturnValue('existing-session-token')
    render(<App />)
    expect(await screen.findByText('Drop an image here or click to select')).toBeInTheDocument()

    await act(async () => {
      window.dispatchEvent(new CustomEvent('truthlens:auth-expired'))
    })
    expect(await screen.findByText('Sign in to your workspace')).toBeInTheDocument()
  })

  it('renders versioned fusion dispositions and unavailable stages without probability percentages', async () => {
    window.history.replaceState({}, '', `/?analysis=${analysisId}`)
    api.getAuthMode.mockResolvedValue({ auth_required: false })
    api.getAnalysisStatus.mockResolvedValue({ analysis_id: analysisId, status: 'partial' })
    api.getAnalysisReport.mockResolvedValue({
      analysis_id: analysisId,
      status: 'partial',
      summary: {},
      stages: { provenance: { status: 'unavailable' } },
      fusion: {
        status: 'completed',
        schema_version: 'truthlens.fusion.v1',
        policy_version: 'e11.conservative.v1',
        category: 'inconclusive',
        overall_score: 0.9,
        confidence: 0.8,
        assessment_confidence: null,
        uncertainty: { status: 'unquantified' },
        informational_findings: [{
          evidence_id: 'ocr-1', source: 'tesseract', status: 'completed', outcome: 'no_text',
          description: 'No text detected in the image.',
        }],
        evidence_excluded: [{ evidence_id: 'prov-1', source: 'google_vision_web_detection', status: 'unavailable', reason: 'stage_unavailable' }],
        missing_evidence: [{ stage_name: 'provenance', status: 'unavailable', reason: 'No usable evidence; absence is not negative evidence.' }],
        stage_execution_summary: { provenance: { status: 'unavailable' } },
        limitations: ['Third-party and contextual evidence is informational.'],
      },
      authenticity: { model_prediction: 'likely_ai_generated', ai_generated_score: 0.9, authentic_score: 0.1 },
    })

    render(<App />)
    expect(await screen.findByText('Final evidence assessment')).toBeInTheDocument()
    expect(screen.getByText('Inconclusive')).toBeInTheDocument()
    expect(screen.getByText('Informational findings')).toBeInTheDocument()
    expect(screen.getByText('Missing or unavailable evidence')).toBeInTheDocument()
    expect(screen.getAllByText('Unavailable').length).toBeGreaterThan(0)
    expect(screen.getByText('Uncertainty is unquantified. No displayed score is a calibrated probability.')).toBeInTheDocument()
    expect(screen.getByText('Primary model output')).toBeInTheDocument()
    expect(screen.queryByText('90%')).not.toBeInTheDocument()
  })
})
