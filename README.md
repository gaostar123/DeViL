<div align="center" style="display:flex; align-items:center; justify-content:center; gap:16px;">
  <img src="assets/logo.png" alt="DeViL logo" width="85">
  <div align="left">
    <h1 style="margin:0;">DeViL</h1>
    <h2 style="margin:4px 0 0;">1+1 > 2 : Detector-Empowered Video Large Language Model for Spatio-Temporal Grounding and Reasoning</h2>
  </div>
</div>

<div align="center">

[**Shida Gao**](https://github.com/gaostar123) · [**Feng Xue**](https://scholar.google.com/citations?user=66SeiQsAAAAJ)<sup>*</sup> · [**Xiangfeng Wang**](https://github.com/sweetsora233)<sup>*</sup> · [**Anlong Ming**](https://scholar.google.com/citations?user=y5kFLCwAAAAJ&hl=en)<sup>✉</sup><br>
[**Teng Long**](https://scholar.google.com/citations?user=5Iv3ul0AAAAJ&hl=en) · [**Yihua Shao**](https://scholar.google.cz/citations?user=lH6YmOUAAAAJ&hl=zh-CN) · [**Haozhe Wang**](https://scholar.google.com/citations?hl=zh-CN&user=V96YGIMAAAAJ) · [**Zhaowen Lin**](https://teacher.bupt.edu.cn/linzhaowen/zh_CN/index.htm) · [**Wei Wang**](https://www.linkedin.com/authwall?trk=bf&trkInfo=AQFIFp8Qre3JwwAAAZrx02d4KQguZTBtrJhFt03M85mSxLu5O0g15lpMNPQ-0kYdcuVhLBwDxGCJq06cFl-rfrAaStunjatHaYbOdO-PppYbom7iGnsK4-4xhaoE5k_TEdp7ioo=&original_referer=&sessionRedirect=https%3A%2F%2Fwww.linkedin.com%2Fin%2F%25E4%25BC%259F-%25E7%258E%258B-ba9945376%3Ftrk%3Dcontact-info) · [**Nicu Sebe**](https://scholar.google.it/citations?user=stFCYOAAAAAJ&hl=en)

<sup>*</sup>Equal contribution. <sup>✉</sup>Corresponding author.

<a href='#'><img src='https://img.shields.io/badge/Project-Page-Green' alt='Project Page'></a>
<a href='http://arxiv.org/abs/2512.06673'><img src='https://img.shields.io/badge/Paper-PDF-orange' alt='Paper PDF'></a>
</div>
<p align="center"><i>This work presents DeViL, a detector-empowered MLLM that mitigates error accumulation and exposure bias seen in previous methods. Compared with previous methods, it delivers faster inference, fewer parameters, and higher accuracy.</i></p>

<p align="center">
<img src="assets/compare.png" style="width: ">
</p>


## 📰 News

- 2025-12-09 Our paper is now publicly available on [arXiv](http://arxiv.org/abs/2512.06673).


## 📝 Abstract
Spatio-temporal grounding and reasoning aims to locate the temporal segment and spatial region of an event in a video given a user query, while also reasoning about semantics such as causality, temporal order, and action relationships. To achieve this, current MLLMs primarily treats bounding boxes as text tokens and generates them autoregressively. However, such autoregressive spatial decoding leads to very-long output sequences, causing spatial errors to accumulated over time and the localization results to progressively drift across a video. To address this, we present a Detector-Empowered Video LLM, short for <strong>DEViL</strong>, which couples a Video LLM with an open-vocabulary detector (OVD). Specifically, the MLLM and detector are connected via a reference-semantic token (RST) that distills the user query into a rich semantic representation. Unlike tokens that merely serve as spatial prompts or segmentor switches, the RST functions as both a control signal and a replacement for the OVD's text embedding, enabling end-to-end learning of both referential understanding and spatial localization. Furthermore, we propose a tube-mined temporal regularization (TTReg) within OVD, which drives the OVD to generate temporally-consistent queries for target objects, thereby ensuring effective temporal association. Experiments demonstrate that DEViL achieves strong performance across various fine-grained video understanding tasks, particularly STVG and GroundedVQA.

## 🔎 Framework
![model](assets/pipline.png)


# Citation

If you use our work or our implementation in this repo, or find them helpful, please consider giving a citation in the following format.

```
@article{gao20251+,
  title={1+ 1> 2: Detector-Empowered Video Large Language Model for Spatio-Temporal Grounding and Reasoning},
  author={Gao, Shida and Xue, Feng and Wang, Xiangfeng and Ming, Anlong and Long, Teng and Shao, Yihua and Wang, Haozhe and Lin, Zhaowen and Wang, Wei and Sebe, Nicu},
  journal={arXiv preprint arXiv:2512.06673},
  year={2025}
}

```

# Acknowledgements

We sincerely thank the following projects for their contributions to this work:

- [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 
