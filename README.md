# TruthLens

**AI-Assisted Image Verification, Analysis and Virality Estimation Including Social Impact**

> ⚠️ TruthLens is an AI-assisted verification system, NOT an absolute truth detector.
> No model prediction should ever be cited as proof that an image is fake or real.

---

## Overview

TruthLens takes an image as input and produces a comprehensive analysis covering:

- 🔍 **Authenticity Assessment** — Is the image likely AI-generated, manipulated, or authentic? (probabilistic, multi-model ensemble)
- 🧩 **Manipulation Detection** — Splicing, copy-move, and inpainting indicators
- 🔬 **Visual Forensics** — Error Level Analysis (ELA), noise patterns, JPEG artifact maps
- 📝 **OCR & Text Analysis** — Extract text, detect language, extract claims
- 🧠 **Contextual Analysis** — Vision-language model reasoning about scene consistency
- 🗺️ **Explainable AI** — Grad-CAM heatmaps showing which regions influenced the AI decision
- 🔗 **Provenance Search** — Reverse image lookup for source attribution
- ✅ **Fact-Checking** — Automated claim verification against fact-check databases
- 📊 **Virality Estimation** — Spread risk assessment (heuristic-based)
- 🇮🇳 **Regional Spread Risk** — India-specific geographic risk map
- ⚖️ **Social Impact** — Potential societal impact indicators

---

## Technology Stack

| Layer     | Technologies                                                    |
| --------- | --------------------------------------------------------------- |
| Frontend  | React, TypeScript, Vite, Tailwind CSS, Recharts, Leaflet       |
| Backend   | Python, FastAPI, Pydantic, SQLAlchemy, Uvicorn                  |
| ML/AI     | PyTorch, Hugging Face Transformers, timm, EasyOCR, Ultralytics |
| Database  | PostgreSQL (Docker) / SQLite (fallback)                         |
| Infra     | Docker, Docker Compose                                          |

---

## System Requirements

| Component | Minimum                                        |
| --------- | ---------------------------------------------- |
| CPU       | 4+ cores (Intel i5 / AMD Ryzen 5 or better)   |
| RAM       | 16 GB                                          |
| GPU       | NVIDIA GPU with 4+ GB VRAM (CUDA 11.8+)       |
| Storage   | 10 GB free (for model weights and artifacts)   |
| Python    | 3.12+                                          |
| Node.js   | 18+                                            |
| Docker    | 20+ (for PostgreSQL)                           |

> GPU is optional — the system falls back to CPU with longer inference times.

---

## Quick Start

### 1. Clone and Configure

```bash
cd "final year project"
cp .env.example .env
# Edit .env with your configuration
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Start Database (Docker)

```bash
docker run -d --name truthlens-pg \
  -e POSTGRES_PASSWORD=truthlens \
  -e POSTGRES_DB=truthlens \
  -p 5432:5432 \
  postgres:16
```

### 4. Start Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 5. Install & Start Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Project Structure

```
truthlens/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── api/       # API endpoints
│   │   ├── core/      # Security, logging
│   │   ├── db/        # Database setup
│   │   ├── models/    # SQLAlchemy ORM models
│   │   ├── schemas/   # Pydantic schemas
│   │   ├── services/  # Business logic
│   │   └── utils/     # Utilities
│   └── tests/
├── frontend/          # React frontend
├── models/            # ML model configs & cache
├── storage/           # Uploads & artifacts
├── docs/              # Documentation
├── docker/            # Docker configuration
└── scripts/           # Setup & utility scripts
```

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System architecture and data flow
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) — Staged development roadmap
- [Model Registry](docs/MODELS.md) — All ML models with specifications

---

## API Endpoints

| Method | Endpoint                              | Description                    |
| ------ | ------------------------------------- | ------------------------------ |
| POST   | `/api/v1/analyze`                     | Submit image for analysis      |
| GET    | `/api/v1/analysis/{id}`               | Get analysis results           |
| GET    | `/api/v1/analysis/{id}/report`        | Get formatted report           |
| GET    | `/api/v1/analysis/{id}/heatmap`       | Get heatmap overlay            |
| POST   | `/api/v1/ocr`                         | Standalone OCR                 |
| POST   | `/api/v1/fact-check`                  | Standalone fact-checking       |
| POST   | `/api/v1/provenance`                  | Reverse image search           |
| GET    | `/api/v1/models`                      | List available models          |
| GET    | `/api/v1/health`                      | Health check                   |
| GET    | `/api/v1/health/models`               | Model health status            |

---

## ML Models Used

| Purpose                   | Model                                           | Parameters |
| ------------------------- | ----------------------------------------------- | ---------- |
| Authenticity Detection    | `prithivMLmods/Deepfake-Real-Class-Siglip2`     | ~86M       |
| Forensic Detection        | `buildborderless/CommunityForensics-DeepfakeDet-ViT` | ~22M  |
| Context Analysis (VLM)    | `Qwen/Qwen2.5-VL-3B-Instruct`                  | ~3B        |
| Language Detection        | `papluca/xlm-roberta-base-language-detection`   | ~278M      |
| Object Detection          | Ultralytics YOLO26s                             | ~9.5M      |
| OCR                       | EasyOCR                                         | Varies     |

> See [docs/MODELS.md](docs/MODELS.md) for complete model specifications, licenses, and VRAM requirements.

---

## Important Disclaimers

1. **Not a truth detector**: TruthLens provides AI-assisted probabilistic assessments. Model predictions can be wrong.
2. **Concept drift**: AI-generated content detection is an evolving field. Models may become less accurate as generation technology improves.
3. **Data transparency**: Virality estimation, regional spread, and social impact modules use heuristic methods due to lack of empirical training data. These outputs are clearly labeled as such.
4. **Research context**: This system is designed for research and educational purposes.

---

## License

This project is developed as a final year research project.

---

## Acknowledgments

- Hugging Face for model hosting and the Transformers library
- The CommunityForensics project for large-scale deepfake detection research
- Alibaba Cloud for the Qwen2.5-VL model family
- Ultralytics for the YOLO family
- The open-source AI/ML community
