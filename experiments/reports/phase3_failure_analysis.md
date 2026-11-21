# TruthLens Phase 3: Comprehensive Failure Analysis Report

## 1. Executive Summary of Failure Modes

This report investigates error patterns of the frozen SigLIP authenticity detector across the held-out test splits.
We identify high-confidence false positives (authentic images misclassified as AI), high-confidence false negatives
(AI-generated images misclassified as authentic), and borderline ambiguous cases.

---
## 2. Dataset: TINY_GENIMAGE
- **Held-out Test Samples**: 7000
- **False Positives**: 2384 (34.06%)
- **False Negatives**: 753 (10.76%)

### Top High-Confidence False Positives (Authentic Misclassified as AI)
| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator/Source | Group ID |
|---|---|---|---|---|---|
| `imagenet_glide/val/nature/ILSVRC2012_val_00027396.JPEG` | `likely_authentic` | `0.6325` | `7.6238` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |
| `imagenet_midjourney/val/nature/ILSVRC2012_val_00013069.JPEG` | `likely_authentic` | `0.6303` | `7.4892` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |
| `imagenet_ai_0508_adm/val/nature/ILSVRC2012_val_00029152.JPEG` | `likely_authentic` | `0.6224` | `7.0148` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |
| `imagenet_ai_0508_adm/val/nature/ILSVRC2012_val_00034156.JPEG` | `likely_authentic` | `0.6174` | `6.7192` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |
| `imagenet_ai_0419_vqdm/val/nature/ILSVRC2012_val_00006272.JPEG` | `likely_authentic` | `0.6141` | `6.5216` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |
| `imagenet_glide/val/nature/ILSVRC2012_val_00017147.JPEG` | `likely_authentic` | `0.6137` | `6.4997` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |
| `imagenet_ai_0508_adm/val/nature/ILSVRC2012_val_00003846.JPEG` | `likely_authentic` | `0.6112` | `6.3510` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |
| `imagenet_ai_0508_adm/val/nature/ILSVRC2012_val_00041900.JPEG` | `likely_authentic` | `0.6111` | `6.3426` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |

### Top High-Confidence False Negatives (AI Misclassified as Authentic)
| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator | Group ID |
|---|---|---|---|---|---|
| `imagenet_midjourney/val/ai/222_midjourney_197.png` | `ai_generated` | `0.4078` | `-5.2364` | `Midjourney` | `tiny_genimage_val_Midjourney_222` |
| `imagenet_ai_0419_biggan/val/ai/725_biggan_00035.png` | `ai_generated` | `0.4160` | `-4.7616` | `BigGAN` | `tiny_genimage_val_BigGAN_725` |
| `imagenet_ai_0508_adm/val/ai/614_adm_153.PNG` | `ai_generated` | `0.4182` | `-4.6373` | `ADM` | `tiny_genimage_val_ADM_614` |
| `imagenet_ai_0419_vqdm/val/ai/VQDM_1000_200_05_591_vqdm_00035.png` | `ai_generated` | `0.4210` | `-4.4743` | `VQDM` | `tiny_genimage_val_VQDM_VQDM` |
| `imagenet_ai_0419_biggan/val/ai/051_biggan_00039.png` | `ai_generated` | `0.4232` | `-4.3479` | `BigGAN` | `tiny_genimage_val_BigGAN_051` |
| `imagenet_ai_0419_biggan/val/ai/431_biggan_00074.png` | `ai_generated` | `0.4243` | `-4.2819` | `BigGAN` | `tiny_genimage_val_BigGAN_431` |
| `imagenet_ai_0419_vqdm/val/ai/VQDM_1000_200_04_411_vqdm_00020.png` | `ai_generated` | `0.4251` | `-4.2380` | `VQDM` | `tiny_genimage_val_VQDM_VQDM` |
| `imagenet_ai_0419_vqdm/val/ai/VQDM_1000_200_09_903_vqdm_00143.png` | `ai_generated` | `0.4274` | `-4.1055` | `VQDM` | `tiny_genimage_val_VQDM_VQDM` |

### Borderline / Most Ambiguous Samples (|p - 0.5| nearest 0)
| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator | Group ID |
|---|---|---|---|---|---|
| `imagenet_ai_0508_adm/val/ai/526_adm_153.PNG` | `ai_generated` | `0.5000` | `0.0004` | `ADM` | `tiny_genimage_val_ADM_526` |
| `imagenet_ai_0424_wukong/val/ai/828_wukong_image86.png` | `ai_generated` | `0.5000` | `0.0004` | `Wukong` | `tiny_genimage_val_Wukong_828` |
| `imagenet_ai_0419_biggan/val/ai/093_biggan_00127.png` | `ai_generated` | `0.5000` | `-0.0008` | `BigGAN` | `tiny_genimage_val_BigGAN_093` |
| `imagenet_midjourney/val/nature/ILSVRC2012_val_00036648.JPEG` | `likely_authentic` | `0.5000` | `0.0011` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |
| `imagenet_midjourney/val/nature/ILSVRC2012_val_00027765.JPEG` | `likely_authentic` | `0.5000` | `-0.0012` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |
| `imagenet_midjourney/val/nature/ILSVRC2012_val_00025239.JPEG` | `likely_authentic` | `0.5000` | `0.0021` | `ImageNet-Camera` | `tiny_genimage_val_ImageNet-Camera_ILSVRC2012` |

---
## 2. Dataset: FRONTIER_BENCHMARK_2026_10
- **Held-out Test Samples**: 2091
- **False Positives**: 707 (33.81%)
- **False Negatives**: 171 (8.18%)

### Top High-Confidence False Positives (Authentic Misclassified as AI)
| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator/Source | Group ID |
|---|---|---|---|---|---|
| `images_social/reshared/ia0f421f4adf4.jpg` | `likely_authentic` | `0.8097` | `6.7526` | `Camera/Real` | `frontier_ia0f421f4adf4` |
| `images/if117e797a7e5.jpg` | `likely_authentic` | `0.8090` | `6.7337` | `Camera/Real` | `frontier_if117e797a7e5` |
| `images_social/whatsapp/ia0f421f4adf4.jpg` | `likely_authentic` | `0.8085` | `6.7176` | `Camera/Real` | `frontier_ia0f421f4adf4` |
| `images/ia0f421f4adf4.jpg` | `likely_authentic` | `0.8074` | `6.6852` | `Camera/Real` | `frontier_ia0f421f4adf4` |
| `images_social/reshared/if117e797a7e5.jpg` | `likely_authentic` | `0.8015` | `6.5092` | `Camera/Real` | `frontier_if117e797a7e5` |
| `images_social/whatsapp/if117e797a7e5.jpg` | `likely_authentic` | `0.8002` | `6.4713` | `Camera/Real` | `frontier_if117e797a7e5` |
| `images_social/instagram/if117e797a7e5.jpg` | `likely_authentic` | `0.7985` | `6.4228` | `Camera/Real` | `frontier_if117e797a7e5` |
| `images_social/instagram/ia0f421f4adf4.jpg` | `likely_authentic` | `0.7961` | `6.3532` | `Camera/Real` | `frontier_ia0f421f4adf4` |

### Top High-Confidence False Negatives (AI Misclassified as Authentic)
| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator | Group ID |
|---|---|---|---|---|---|
| `images_social/whatsapp/i9c78518b16bb.jpg` | `ai_generated` | `0.3155` | `-3.6133` | `openai/gpt-image-2` | `frontier_i9c78518b16bb` |
| `images_social/reshared/i9c78518b16bb.jpg` | `ai_generated` | `0.3197` | `-3.5228` | `openai/gpt-image-2` | `frontier_i9c78518b16bb` |
| `images_social/instagram/i9c78518b16bb.jpg` | `ai_generated` | `0.3263` | `-3.3810` | `openai/gpt-image-2` | `frontier_i9c78518b16bb` |
| `images/i9c78518b16bb.png` | `ai_generated` | `0.3279` | `-3.3469` | `openai/gpt-image-2` | `frontier_i9c78518b16bb` |
| `images_social/whatsapp/i63989072c188.jpg` | `ai_generated` | `0.3499` | `-2.8888` | `openai/gpt-image-2` | `frontier_i63989072c188` |
| `images_social/reshared/i63989072c188.jpg` | `ai_generated` | `0.3553` | `-2.7794` | `openai/gpt-image-2` | `frontier_i63989072c188` |
| `images_social/instagram/i63989072c188.jpg` | `ai_generated` | `0.3585` | `-2.7130` | `openai/gpt-image-2` | `frontier_i63989072c188` |
| `images_social/reshared/i01af8643156c.jpg` | `ai_generated` | `0.3663` | `-2.5569` | `flux-2` | `frontier_i01af8643156c` |

### Borderline / Most Ambiguous Samples (|p - 0.5| nearest 0)
| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator | Group ID |
|---|---|---|---|---|---|
| `images_social/whatsapp/i2a0a80afc26c.jpg` | `likely_authentic` | `0.4998` | `-0.0034` | `Camera/Real` | `frontier_i2a0a80afc26c` |
| `images_social/instagram/i57362d9a2f74.jpg` | `ai_generated` | `0.5003` | `0.0052` | `luma/agent/uni-1/v1` | `frontier_i57362d9a2f74` |
| `images_social/whatsapp/id4f2d066d9ff.jpg` | `likely_authentic` | `0.4996` | `-0.0075` | `Camera/Real` | `frontier_id4f2d066d9ff` |
| `images_social/instagram/iac91ba464d83.jpg` | `likely_authentic` | `0.4996` | `-0.0080` | `Camera/Real` | `frontier_iac91ba464d83` |
| `images_social/instagram/ic5c88bf915f5.jpg` | `ai_generated` | `0.5005` | `0.0084` | `xai/grok-imagine-image/v2.0` | `frontier_ic5c88bf915f5` |
| `images_social/instagram/i1d42446fcfd8.jpg` | `ai_generated` | `0.5005` | `0.0101` | `ideogram/v4.5` | `frontier_i1d42446fcfd8` |

---
## 2. Dataset: WILDAIGC_BENCH
- **Held-out Test Samples**: 1294
- **False Positives**: 285 (22.02%)
- **False Negatives**: 428 (33.08%)

### Top High-Confidence False Positives (Authentic Misclassified as AI)
| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator/Source | Group ID |
|---|---|---|---|---|---|
| `raw/img_5819c1866d4cda67.webp` | `likely_authentic` | `0.5089` | `5.2829` | `Documentary/Photojournalism` | `wildaigc_cc_0406` |
| `raw/img_50cb45f485040f65.jpg` | `likely_authentic` | `0.5086` | `5.0900` | `Documentary/Photojournalism` | `wildaigc_cc_0373` |
| `raw/img_a009c2fdf09ef3e7.jpg` | `likely_authentic` | `0.5080` | `4.7384` | `Documentary/Photojournalism` | `wildaigc_cc_0755` |
| `raw/img_98cd1226563620ad.jpg` | `likely_authentic` | `0.5077` | `4.5468` | `Documentary/Photojournalism` | `wildaigc_cc_0722` |
| `raw/img_1fc7b406041152ce.jpg` | `likely_authentic` | `0.5073` | `4.3631` | `Documentary/Photojournalism` | `wildaigc_cc_0148` |
| `raw/img_a25c12e218b7fe27.jpg` | `likely_authentic` | `0.5070` | `4.1712` | `Documentary/Photojournalism` | `wildaigc_cc_0765` |
| `raw/img_7ee857f4c5e13a80.webp` | `likely_authentic` | `0.5070` | `4.1606` | `Documentary/Photojournalism` | `wildaigc_cc_0601` |
| `raw/img_0c2ef9992fc68af6.jpg` | `likely_authentic` | `0.5070` | `4.1367` | `Documentary/Photojournalism` | `wildaigc_cc_0055` |

### Top High-Confidence False Negatives (AI Misclassified as Authentic)
| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator | Group ID |
|---|---|---|---|---|---|
| `processed/img_de0a73255e9d9867.jpg` | `ai_generated` | `0.4913` | `-5.1919` | `Wild-AI-Generated` | `wildaigc_cc_1042` |
| `raw/img_de0a73255e9d9867.jpg` | `ai_generated` | `0.4913` | `-5.1919` | `Wild-AI-Generated` | `wildaigc_cc_1042` |
| `processed/img_68894fdbb2bd9524.jpg` | `ai_generated` | `0.4928` | `-4.2903` | `Wild-AI-Generated` | `wildaigc_cc_0492` |
| `raw/img_68894fdbb2bd9524.jpg` | `ai_generated` | `0.4928` | `-4.2903` | `Wild-AI-Generated` | `wildaigc_cc_0492` |
| `raw/img_20efc2485b765c7d.png` | `ai_generated` | `0.4931` | `-4.0909` | `Wild-AI-Generated` | `wildaigc_cc_0156` |
| `processed/img_389d5651c4658339.jpg` | `ai_generated` | `0.4935` | `-3.8525` | `Wild-AI-Generated` | `wildaigc_cc_0259` |
| `raw/img_389d5651c4658339.jpg` | `ai_generated` | `0.4935` | `-3.8525` | `Wild-AI-Generated` | `wildaigc_cc_0259` |
| `processed/img_eef641f81db6498b.jpg` | `ai_generated` | `0.4935` | `-3.8329` | `Wild-AI-Generated` | `wildaigc_cc_1114` |

### Borderline / Most Ambiguous Samples (|p - 0.5| nearest 0)
| Image Path | Ground Truth | Calibrated AI Prob | Margin | Generator | Group ID |
|---|---|---|---|---|---|
| `raw/img_0b3775a476073c20.jpg` | `ai_generated` | `0.5000` | `0.0016` | `Wild-AI-Generated` | `wildaigc_cc_0050` |
| `processed/img_6242657807a3a290.jpg` | `ai_generated` | `0.5000` | `0.0068` | `Wild-AI-Generated` | `wildaigc_cc_0457` |
| `raw/img_6242657807a3a290.jpg` | `ai_generated` | `0.5000` | `0.0068` | `Wild-AI-Generated` | `wildaigc_cc_0457` |
| `processed/img_5c47072ad5c86468.webp` | `ai_generated` | `0.5000` | `-0.0094` | `Wild-AI-Generated` | `wildaigc_cc_0424` |
| `raw/img_5c47072ad5c86468.webp` | `ai_generated` | `0.5000` | `-0.0094` | `Wild-AI-Generated` | `wildaigc_cc_0424` |
| `processed/img_5fb6be9c740abb02.jpg` | `ai_generated` | `0.5000` | `0.0095` | `Wild-AI-Generated` | `wildaigc_cc_0440` |

## 3. Systematic Architectural Vulnerabilities

1. **Frequency Domain Blindspots**: As a patch-based Vision Transformer with 16x16 pixel patches and standard bicubic resizing to 224x224, the SigLIP backbone loses high-frequency Fourier artifacts and periodic resampling grids common in newer diffusion/autoregressive models.
2. **Social Media Compression Degradation**: JPEG recompression (e.g. WhatsApp, Instagram) smooths subtle generation artifacts, pushing AI-generated images toward authentic scores.
3. **Domain Shift & Stylistic Overlap**: Authentic images with heavy studio lighting, high contrast, or digital HDR post-processing frequently trigger false positive AI classifications.
4. **Cross-Architecture Fragility**: Models trained on older diffusion (e.g. SD 1.5, BigGAN) experience sharp accuracy degradation when encountering modern rectified flow architectures (e.g. Flux-3) or large multimodal diffusion (e.g. GPT-Image-2, Grok Imagine 2).