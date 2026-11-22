import { describe, expect, it } from 'vitest'
import { normalizeProtectedAssetUrl } from '../services/api'

const id = '6a522bf5-32b1-4e4a-8b6d-901f77d270ad'

describe('protected artifact URLs', () => {
  it('accepts only same-origin analysis image and artifact routes', () => {
    expect(normalizeProtectedAssetUrl(`/api/v1/analysis/${id}/image`)).toBe(
      `${window.location.origin}/api/v1/analysis/${id}/image`,
    )
    expect(normalizeProtectedAssetUrl(`/api/v1/analysis/${id}/heatmap?variant=raw`)).toContain(
      `/api/v1/analysis/${id}/heatmap?variant=raw`,
    )
  })

  it('rejects an external target and path traversal before sending bearer credentials', () => {
    expect(() => normalizeProtectedAssetUrl(`https://attacker.example/api/v1/analysis/${id}/image`)).toThrow(
      'not a TruthLens analysis artifact',
    )
    expect(() => normalizeProtectedAssetUrl(`/api/v1/analysis/${id}/../../auth/token`)).toThrow(
      'not a TruthLens analysis artifact',
    )
  })
})
