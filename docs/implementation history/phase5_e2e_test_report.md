# Phase 5 end-to-end and regression test report



Date: 2026-11-17. Environment: Windows 11 desktop sandbox, Python 3.11.9, pytest 9.1.1, Node v22 (bundled workspace runtime was used for build), CPU execution; CUDA availability was not measured in this session. A local browser session and temporary local API server were used; external provider credentials were not configured.



## Automated results



| Command | Result |

|---|---|

| `python -m pytest --noconftest -m "not asyncio" backend/tests/test_phase1.py backend/tests/test_phase2.py backend/tests/test_phase3.py backend/tests/test_stage3.py backend/tests/test_stage4.py backend/tests/test_phase4.py backend/tests/test_phase5.py backend/tests/test_stage5.py backend/tests/test_services.py -q` | **165 passed, 10 skipped, 2 deselected**. Skipped tests require opt-in real-model checks; async tests were deselected because Windows asyncio loop startup is blocked in this sandbox. |

| `python -m pytest backend/tests -q` | **184 passed, 10 skipped in 13.05s**. The suite includes API client tests, including upload-to-partial-report plus stored-image retrieval. Ten skips are opt-in model-dependent checks. |

| `npm run build` (`frontend/`, Node 24.22.1 workspace runtime) | Passed after Phase 5 UI changes. Existing CSS import-order and >500 kB bundle warnings. |

| `npm run lint` (`frontend/`) | Passed. |



## Scenario coverage and limitations



| Scenario | Status |

|---|---|

| Upload signature, extension, size, image decoding | Covered by passing API/unit tests |

| Genuine / AI-generated / manipulated ground-truth images | Not validated; no legally sourced and ground-truthed corpus documented in repository |

| Embedded factual claim and multilingual image | Component tests exist; full upload-to-report path not exercised |

| No external evidence / missing credentials | Service behavior implemented as unavailable/no-results; tests are mocked/unit level |

| Partial pipeline failure | Covered by report/stage tests and upload-to-partial-report API integration test |

| Social analytics outputs | Phase 4 tests passed; no prediction claims made |

| Browser upload flow | The UI submitted a preselected 3840×2160 JPEG when the upload area was activated. The local backend completed 14/20 stages and built a report in 40.4 seconds. The file was not a controlled or ground-truthed sample; the report was not visually verified in the browser. No external provider credentials were configured. |



`test_data/fake_news_test.jpg` and existing files beneath `test_data/` have no checked-in source/license/ground truth manifest. They are therefore not presented as legally verified demo samples. Unit and timing tests generate simple images in memory. A diverse legally cleared demonstration set is still required before academic performance evaluation.



Passing tests and one uncontrolled browser run do not establish model accuracy. API upload-to-report, report retrieval and original-image retrieval passed integration tests. The browser caused a local analysis request with an image of unknown origin; the backend completed and built a report, but visible report rendering was not confirmed. Do not treat this image as evaluation data.
