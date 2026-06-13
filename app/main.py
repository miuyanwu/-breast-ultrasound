"""
乳腺超声智能辅助诊断系统 — Streamlit App
  单病例智能工作站 | 高通量批量筛查中心 | 算法技术白皮书
  运行: streamlit run app/main.py
  设计系统: Apple × IBM 融合
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import time
import base64
from PIL import Image

from src.inference.engine import (
    load_models, infer_single, overlay_mask,
    MODEL_REGISTRY, BIRADS_MAP, resolve_inference_models,
)

# V10 分割→分类函数 — 延迟导入，避免 SMP 缺失影响基础功能
_infer_with_seg = None
_load_seg_pipeline = None


def _get_v10_functions():
    """延迟加载 V10 函数。仅在首次调用时导入。"""
    global _infer_with_seg, _load_seg_pipeline
    if _infer_with_seg is not None:
        return _infer_with_seg, _load_seg_pipeline
    try:
        from src.inference.engine import infer_with_seg, load_seg_pipeline
        _infer_with_seg = infer_with_seg
        _load_seg_pipeline = load_seg_pipeline
        return _infer_with_seg, _load_seg_pipeline
    except ImportError as e:
        st.error(f"""
        **V10 分割→分类模式不可用**

        请使用 conda 环境运行:

        ```
        D:\\Anaconda\\envs\\pytorch\\python.exe -m streamlit run app/main.py
        ```

        原始错误: `{e}`
        """)
        return None, None
from app.report import generate_report_html, generate_report_pdf

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="乳腺超声智能辅助诊断系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

# ============================================================
# 设计系统: Apple × IBM 融合
# ============================================================
CSS_DESIGN_TOKENS = """
/* ── IBM Plex Sans 加载 ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    /* IBM Blue 单一强调色 */
    --color-accent: #4589ff;
    --color-accent-hover: #2563eb;
    --color-accent-light: #edf5ff;

    /* IBM 语义色 */
    --color-danger: #da1e28;
    --color-danger-light: #fff1f1;
    --color-safe: #24a148;
    --color-safe-light: #f1f8f4;

    /* Apple 纯白底色 */
    --color-canvas: #ffffff;
    /* IBM 浅灰面 */
    --color-surface: #f4f4f4;
    /* 深灰影像灯箱 */
    --color-lightbox: #1d1d1f;

    /* 文字层级 */
    --color-ink: #161616;
    --color-ink-muted: #525252;
    --color-ink-subtle: #8c8c8c;
    /* Apple hairline */
    --color-hairline: #e0e0e0;

    /* IBM 4px 间距栅格 */
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 24px;
    --space-6: 32px;
    --space-7: 48px;
    --space-8: 96px;

    /* 字体 */
    --font-sans: 'IBM Plex Sans', 'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', sans-serif;
    --font-mono: 'IBM Plex Mono', 'Consolas', 'Courier New', monospace;

    /* 圆角: 融合 2-4px */
    --radius-sm: 2px;
    --radius-md: 4px;

    /* 过渡 */
    --transition-fast: 150ms ease;
    --transition-normal: 250ms ease;
}

/* ── 全局重置 ── */
body {
    font-family: var(--font-sans);
    font-size: 14px;
    font-weight: 400;
    line-height: 1.6;
    color: var(--color-ink);
    background: var(--color-canvas);
    -webkit-font-smoothing: antialiased;
}

/* Streamlit 全局覆盖 */
.stApp {
    background: var(--color-canvas);
}
/* 强制所有文字为深色 */
.stApp p, .stApp span, .stApp label, .stApp div, .stApp li, .stApp td, .stApp th,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stText"],
.stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown span,
.stApp .st-emotion-cache, .stApp section, .stApp main,
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricDelta"] {
    color: var(--color-ink);
}
/* caption 和 subtle 文字 */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {
    color: var(--color-ink-muted) !important;
}
.stApp [data-testid="stHeader"] {
    background: var(--color-canvas);
    border-bottom: 1px solid var(--color-hairline);
}
section[data-testid="stSidebar"] {
    background: var(--color-surface);
    border-right: 1px solid var(--color-hairline);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-weight: 300 !important;
    font-size: 20px !important;
    letter-spacing: -0.3px;
}

/* ── 按钮 ── */
.stButton > button {
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    border-radius: 0 !important;
    border: 1px solid var(--color-hairline) !important;
    background: var(--color-canvas) !important;
    color: var(--color-ink) !important;
    padding: 8px 20px !important;
    transition: all var(--transition-fast) !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: var(--color-accent) !important;
    color: var(--color-accent) !important;
    background: var(--color-accent-light) !important;
}
.stButton > button:active {
    transform: scale(0.98);
}
/* 主按钮 */
.stButton > button[kind="primary"] {
    background: var(--color-accent) !important;
    color: #ffffff !important;
    border-color: var(--color-accent) !important;
    font-weight: 500 !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--color-accent-hover) !important;
    border-color: var(--color-accent-hover) !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span {
    color: #ffffff !important;
}

/* ── 文件上传区 ── */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--color-hairline) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--color-surface) !important;
    transition: border-color var(--transition-fast);
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--color-accent) !important;
}

/* ── 滑块 ── */
[data-testid="stSlider"] .stSliderTrack {
    background: var(--color-hairline);
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    background: var(--color-accent) !important;
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
}

/* ── 进度条 ── */
[data-testid="stProgress"] > div > div {
    background: var(--color-accent) !important;
    border-radius: 0 !important;
    transition: width 0.3s ease;
}

/* ── Selectbox / Radio ── */
[data-testid="stSelectbox"] select,
.stRadio > div {
    font-family: var(--font-sans) !important;
}
.stRadio [role="radiogroup"] label {
    border: 1px solid var(--color-hairline) !important;
    border-radius: 0 !important;
    padding: 6px 18px !important;
    transition: all var(--transition-fast);
    font-weight: 400 !important;
}
.stRadio [role="radiogroup"] label:hover {
    border-color: var(--color-accent) !important;
    color: var(--color-accent) !important;
}
.stRadio [role="radiogroup"] label[data-selected="true"] {
    background: var(--color-ink) !important;
    color: #ffffff !important;
    border-color: var(--color-ink) !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    border: 1px solid var(--color-hairline);
    border-radius: var(--radius-sm);
    padding: 16px !important;
    background: var(--color-canvas);
}
[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 400 !important;
    color: var(--color-ink-subtle) !important;
    text-transform: uppercase;
    letter-spacing: 0.32px;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 300 !important;
    letter-spacing: -0.5px;
    color: var(--color-ink) !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--color-hairline) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stDataFrame"] th {
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    color: var(--color-ink-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.32px;
    background: var(--color-surface) !important;
    border-bottom: 1px solid var(--color-hairline) !important;
}

/* ── 卡片 ── */
.card {
    border: 1px solid var(--color-hairline);
    border-radius: var(--radius-sm);
    padding: var(--space-5);
    background: var(--color-canvas);
    transition: border-color var(--transition-fast);
}
.card:hover {
    border-color: var(--color-accent);
}
.card-fill {
    min-height: 100%;
}

/* ── 动效: fadeInUp ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeInUp 0.4s ease both;
}
.animate-in:nth-child(1) { animation-delay: 0.05s; }
.animate-in:nth-child(2) { animation-delay: 0.10s; }
.animate-in:nth-child(3) { animation-delay: 0.15s; }
.animate-in:nth-child(4) { animation-delay: 0.20s; }

/* ── 影像灯箱 ── */
.lightbox {
    background: var(--color-lightbox);
    padding: 8px 8px 0 8px;
    border-radius: var(--radius-sm);
    text-align: center;
}
.lightbox img {
    box-shadow: 0 2px 16px rgba(0,0,0,0.30);
    border-radius: var(--radius-sm);
    display: block;
    margin: 0 auto;
}
.lightbox-label {
    color: #c0c0c0;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.32px;
    text-align: center;
    margin: 0 0 6px;
    padding: 6px 0 0;
}

/* ── 时间轴 ── */
.timeline-entry {
    padding: 14px 0 14px 20px;
    border-left: 2px solid var(--color-hairline);
    margin-left: 6px;
    position: relative;
    transition: border-color var(--transition-fast);
}
.timeline-entry:hover { border-left-color: var(--color-accent); }
.timeline-entry::before {
    content: '';
    position: absolute; left: -5px; top: 18px;
    width: 8px; height: 8px;
    background: var(--color-accent);
    border-radius: 0;
}
.timeline-entry h4 {
    margin: 0 0 4px;
    font-size: 15px;
    font-weight: 500;
    color: var(--color-ink);
}
.timeline-entry p {
    margin: 0;
    font-size: 13px;
    color: var(--color-ink-muted);
    line-height: 1.5;
}

/* ── insight 条目 ── */
.insight-item {
    padding: 10px 0;
    border-bottom: 1px solid var(--color-hairline);
    display: flex;
    gap: var(--space-3);
    align-items: baseline;
}
.insight-item:last-child { border-bottom: none; }
.insight-item .tag {
    font-size: 11px;
    font-weight: 500;
    padding: 1px 6px;
    background: var(--color-surface);
    border: 1px solid var(--color-hairline);
    border-radius: 0;
    white-space: nowrap;
    letter-spacing: 0.16px;
}
.insight-item .tag.pos { color: var(--color-safe); border-color: var(--color-safe); }
.insight-item .tag.neg { color: var(--color-danger); border-color: var(--color-danger); }
.insight-item .text { font-size: 13px; color: var(--color-ink); line-height: 1.5; }
.insight-item .text b { font-weight: 500; }

/* ── 下载按钮 ── */
[data-testid="stDownloadButton"] > button {
    border-radius: 0 !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
}

/* ── 首页卡片 ── */
.home-card {
    border: 1px solid var(--color-hairline);
    border-radius: 2px;
    padding: 32px 28px;
    cursor: pointer;
    transition: all var(--transition-fast);
    background: var(--color-canvas);
    min-height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.home-card:hover {
    border-color: var(--color-accent);
    background: var(--color-accent-light);
}
.home-card .card-icon {
    font-size: 28px;
    margin-bottom: 12px;
}
.home-card .card-title {
    font-size: 20px;
    font-weight: 300;
    color: var(--color-ink);
    letter-spacing: -0.3px;
    margin: 0 0 8px;
}
.home-card .card-desc {
    font-size: 13px;
    color: var(--color-ink-muted);
    line-height: 1.5;
    margin: 0;
}
"""

# ============================================================
# 全局 CSS 注入
# ============================================================
st.markdown(f"<style>{CSS_DESIGN_TOKENS}</style>", unsafe_allow_html=True)

# ============================================================
# 页面路由: session_state 首页仪表盘
# ============================================================
if 'page' not in st.session_state:
    st.session_state.page = '首页'

# 非首页显示返回按钮 + 模块切换
module_pages = ["模型效果展示", "单病例智能工作站", "高通量批量筛查中心", "算法技术白皮书"]

if st.session_state.page != '首页':
    col_back, col_switch, _ = st.columns([1, 1.5, 5])
    with col_back:
        if st.button("← 返回首页", key="back_home", use_container_width=True):
            st.session_state.page = '首页'
            st.rerun()
    with col_switch:
        current_idx = module_pages.index(st.session_state.page) if st.session_state.page in module_pages else 0
        new_page = st.selectbox(
            "切换模块",
            module_pages,
            index=current_idx,
            key="top_module_switcher",
            label_visibility="collapsed",
        )
        if new_page != st.session_state.page:
            st.session_state.page = new_page
            st.rerun()

page = st.session_state.page

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 4px;">
        <h2 style="font-weight:300;font-size:20px;letter-spacing:-0.3px;margin:0;color:#161616;">
            乳腺超声<br>智能辅助诊断系统
        </h2>
        <p style="font-size:12px;color:#8c8c8c;margin:6px 0 0;">Breast Ultrasound AI Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    threshold = st.slider(
        "诊断阈值",
        min_value=0.20, max_value=0.80, value=0.45, step=0.05,
        help="滑动调节良/恶性判定边界。阈值越低，灵敏度越高（适合筛查）；阈值越高，特异性越高（适合确诊）。"
    )

    if threshold < 0.45:
        hint_icon, hint_text, hint_desc = (
            "●", "早期筛查模式",
            "高灵敏度 · 宁可误判，不可漏诊"
        )
    elif threshold <= 0.50:
        hint_icon, hint_text, hint_desc = (
            "●", "均衡诊断模式",
            "平衡灵敏度与特异性"
        )
    else:
        hint_icon, hint_text, hint_desc = (
            "●", "确诊模式",
            "高特异性 · 减少假阳性，辅助术前决策"
        )

    st.markdown(f"""
    <div style="padding:12px;margin:8px 0;border:1px solid #e0e0e0;border-left:3px solid #4589ff;border-radius:0;">
        <b style="font-size:13px;">{hint_icon} {hint_text}</b><br>
        <small style="color:#525252;">{hint_desc}</small>
    </div>
    """, unsafe_allow_html=True)

    inference_preset = st.selectbox(
        "推理模式",
        ["seg_then_classify", "compact", "compact_weighted", "ensemble4", "distill", "soup", "v23"],
        index=0,
        format_func=lambda x: {
            "seg_then_classify": "🏆 V10 分割→分类 (AUC 0.9225)",
            "compact": "Compact 自动 (V9)",
            "compact_weighted": "Compact 加权 logit",
            "ensemble4": "4 模型融合对照",
            "distill": "DISTILL 单模型",
            "soup": "SOUP 权重汤单模型",
            "v23": "V2.3 单模型",
        }[x],
        help="🏆 V10 分割→分类: 先用 MiT-B2 UNet 定位病灶，再对原图+ROI分别 V9 分类后 0.5:0.5 融合 (AUC=0.9225, 当前最佳)。推理时间约为 V9 的 3 倍，但敏感度显著提升。",
    )
    active_model_names = resolve_inference_models(inference_preset)

    st.markdown("---")

    st.markdown("##### 模型集群状态")
    for name, cfg in MODEL_REGISTRY.items():
        st.checkbox(cfg['label'], value=name in active_model_names, disabled=True, key=f"model_{name}_{inference_preset}")

    if inference_preset == "seg_then_classify":
        st.markdown("##### 分割管线 (V10)")
        st.checkbox("MiT-B2 UNet 粗分割 ×2", value=True, disabled=True, key="seg_coarse_status")
        st.checkbox("MiT-B2 UNet ROI 精修 ×1", value=True, disabled=True, key="seg_roi_status")
        st.caption("V10 分割→分类: AUC=0.9225, F1@0.55=0.7971")
    else:
        st.caption("当前最佳: V9 DISTILL+SOUP+V2.3 加权 logit, AUC=0.9139, F1@0.55=0.7952")

    st.markdown("---")
    st.caption("© 2026 重庆邮电大学 生物医学工程竞赛团队")

# ============================================================
# 模型预加载
# ============================================================
device_str = 'cuda'
with st.spinner(f"正在启动 {' + '.join(active_model_names)} 推理引擎..."):
    load_models(device_str, inference_preset)
    if inference_preset == "seg_then_classify":
        _, _load_seg = _get_v10_functions()
        if _load_seg:
            _load_seg(device_str)

# ============================================================
# 辅助函数
# ============================================================
def run_inference_and_overlay(uploaded_file_bytes):
    """对单张图执行推理并返回叠加结果。V10 模式自动走分割→分类管线。"""
    if inference_preset == "seg_then_classify":
        _infer_seg, _ = _get_v10_functions()
        if _infer_seg is None:
            st.error("V10 模式不可用，请使用 conda 环境运行。已回退到 V9 模式。")
            result = infer_single(uploaded_file_bytes, device_str, "compact_weighted")
        else:
            result = _infer_seg(uploaded_file_bytes, device_str, "compact_weighted")
        is_mal = result['prob'] >= threshold
        result['prediction'] = '恶性' if is_mal else '良性'
        # V10 已经包含了 overlay (用分割掩膜)
        if 'image_overlay' in result:
            result['overlay'] = result['image_overlay']
        else:
            result['overlay'] = overlay_mask(
                result['image_original'], result['mask_binary'], is_mal
            )
        return result
    else:
        result = infer_single(uploaded_file_bytes, device_str, inference_preset)
        is_mal = result['prob'] >= threshold
        result['prediction'] = '恶性' if is_mal else '良性'
        overlay = overlay_mask(
            result['image_original'], result['mask_binary'], is_mal
        )
        result['overlay'] = overlay
        return result


# ── 模型效果展示: 缓存函数 (模块级) ──
@st.cache_data(show_spinner=False)
def _get_demo_pool_paths(pool_size=30):
    """从测试集获取一批样本路径用于筛选。

    优先使用 data/测试集/（本地开发），若不存在则回退到
    app/demo_samples/（HF Spaces / 精简部署）。
    """
    import glob as _glob
    benign_dir = os.path.join(_PROJECT_ROOT, 'data', '测试集', 'benign')
    malignant_dir = os.path.join(_PROJECT_ROOT, 'data', '测试集', 'malignant')

    if not os.path.isdir(benign_dir) or not os.path.isdir(malignant_dir):
        # 回退到内置 demo 样本
        benign_dir = os.path.join(_PROJECT_ROOT, 'app', 'demo_samples', 'benign')
        malignant_dir = os.path.join(_PROJECT_ROOT, 'app', 'demo_samples', 'malignant')

    b_imgs = sorted([f for f in _glob.glob(os.path.join(benign_dir, '*.png'))
                    if '_mask' not in os.path.basename(f)])
    m_imgs = sorted([f for f in _glob.glob(os.path.join(malignant_dir, '*.png'))
                    if '_mask' not in os.path.basename(f)])
    if not b_imgs or not m_imgs:
        return [], []

    import random as _random
    _random.seed(42)
    actual_pool = min(pool_size, len(b_imgs), len(m_imgs))
    b_pool = _random.sample(b_imgs, actual_pool)
    m_pool = _random.sample(m_imgs, actual_pool)
    return b_pool, m_pool


@st.cache_data(show_spinner="正在运行模型推理...")
def _demo_inference_curated(b_pool_tuple, m_pool_tuple, preset, threshold,
                            n_good=3, n_borderline=1):
    """从候选池中推理并精选高质量样本: 按置信度排序, 每类选 n_good 个高置信度 + n_borderline 个边界样本"""
    results = []
    # 推理所有候选
    for path in list(b_pool_tuple) + list(m_pool_tuple):
        with open(path, 'rb') as f:
            img_bytes = f.read()
        r = infer_single(img_bytes, device_str, preset)
        is_mal = r['prob'] >= threshold
        r['prediction'] = '恶性' if is_mal else '良性'
        r['filepath'] = path
        gt_path = path.replace('.png', '_mask.png')
        if os.path.exists(gt_path):
            gt_mask = np.array(Image.open(gt_path).resize((256, 256)).convert('L')) > 128
        else:
            gt_mask = np.zeros((256, 256), dtype=bool)
        r['gt_mask'] = gt_mask
        r['image_np'] = np.array(Image.open(path).resize((256, 256)).convert('RGB'))
        results.append(r)
    # 按是否真的恶性分组
    b_res = [r for r in results if 'benign' in os.path.basename(os.path.dirname(r['filepath']))]
    m_res = [r for r in results if 'malignant' in os.path.basename(os.path.dirname(r['filepath']))]
    if not b_res:
        b_res = [r for r in results if r['prob'] < 0.5]
    if not m_res:
        m_res = [r for r in results if r['prob'] >= 0.5]
    # 良性: 按 prob 升序 (越接近 0 越自信)
    b_sorted = sorted(b_res, key=lambda x: x['prob'])
    # 恶性: 按 prob 降序 (越接近 1 越自信)
    m_sorted = sorted(m_res, key=lambda x: x['prob'], reverse=True)
    # 精选: n_good 个最佳 + n_borderline 个边界 (prob 接近 0.5)
    b_borderline = sorted(b_res, key=lambda x: abs(x['prob'] - 0.5))
    m_borderline = sorted(m_res, key=lambda x: abs(x['prob'] - 0.5))
    b_selected = b_sorted[:n_good] + b_borderline[:n_borderline]
    m_selected = m_sorted[:n_good] + m_borderline[:n_borderline]
    # 去重
    seen = set()
    b_final, m_final = [], []
    for r in b_selected:
        if id(r) not in seen:
            seen.add(id(r))
            b_final.append(r)
    for r in m_selected:
        if id(r) not in seen:
            seen.add(id(r))
            m_final.append(r)
    return b_final[:n_good + n_borderline], m_final[:n_good + n_borderline]




# ============================================================
# 首页
# ============================================================
if page == "首页":
    st.markdown("""
    <div style="text-align:center;padding:48px 0 32px;">
        <h1 style="font-weight:300;font-size:36px;letter-spacing:-0.5px;color:#161616;margin:0 0 8px;">
            乳腺超声智能辅助诊断系统
        </h1>
        <p style="font-size:15px;color:#525252;margin:0;">
            Breast Ultrasound AI-Assisted Diagnosis System
        </p>
    </div>
    """, unsafe_allow_html=True)

    card_data = [
        ("模型效果展示", "测试集样本分割与分类效果对比，直观评估模型泛化能力",
         "d0"),
        ("单病例智能工作站", "上传单张超声图像，AI 进行病灶分割与良恶性诊断，一键生成 PDF 报告",
         "d1"),
        ("高通量批量筛查中心", "批量上传多张图像，AI 快速筛查并生成统计报表与 CSV 导出",
         "d2"),
        ("算法技术白皮书", "35 实验完整记录 · V10 分割→分类 · 性能对比与核心经验",
         "d3"),
    ]

    row1_cols = st.columns(2)
    for i, (title, desc, key) in enumerate(card_data[:2]):
        with row1_cols[i]:
            st.markdown(f"""
            <div class="home-card">
                <div class="card-title">{title}</div>
                <div class="card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("进入", key=f"home_{key}", use_container_width=True):
                st.session_state.page = title
                st.rerun()

    row2_cols = st.columns(2)
    for i, (title, desc, key) in enumerate(card_data[2:]):
        with row2_cols[i]:
            st.markdown(f"""
            <div class="home-card">
                <div class="card-title">{title}</div>
                <div class="card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("进入", key=f"home_{key}", use_container_width=True):
                st.session_state.page = title
                st.rerun()



# ============================================================
# 页面零: 模型效果展示
# ============================================================
if page == "模型效果展示":
    st.markdown("""
    <h2 style="font-weight:300;font-size:32px;letter-spacing:-0.5px;margin:24px 0 4px;">模型效果展示</h2>
    <p style="color:#525252;font-size:14px;margin:0 0 24px;">测试集样本对比 — 真值掩膜 vs 模型预测掩膜 · 分割 + 分类效果一览</p>
    """, unsafe_allow_html=True)

    b_pool, m_pool = _get_demo_pool_paths()

    if not b_pool and not m_pool:
        st.warning("测试集图像未找到，请确保 data/测试集/ 目录存在。")
    else:
        b_selected, m_selected = _demo_inference_curated(
            tuple(b_pool), tuple(m_pool), inference_preset, threshold,
            n_good=3, n_borderline=1
        )
        demo_results = b_selected + m_selected
        true_labels = (['良性'] * len(b_selected) +
                       ['恶性'] * len(m_selected))

        # 判断当前推理模式是否有 BI-RADS 能力
        has_birads = any(MODEL_REGISTRY.get(n, {}).get('has_birads', False)
                        for n in active_model_names)

        st.markdown("---")

        for idx, (res, true_label) in enumerate(zip(demo_results, true_labels)):
            prob = res['prob']
            pred_label = res['prediction']
            is_mal_true = true_label == '恶性'
            is_mal_pred = pred_label == '恶性'
            correct = (is_mal_true == is_mal_pred)

            # GT overlay
            gt_overlay = overlay_mask(res['image_np'], res['gt_mask'], is_mal_true)
            # Pred overlay
            pred_overlay = overlay_mask(res['image_np'], res['mask_binary'], is_mal_pred)

            col_orig, col_gt, col_pred, col_info = st.columns([1, 1, 1, 0.8])

            with col_orig:
                st.caption(f"原始影像 #{idx+1}")
                st.markdown('<div class="lightbox">', unsafe_allow_html=True)
                st.image(res['image_np'], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_gt:
                st.caption("真值掩膜 (Ground Truth)")
                st.markdown('<div class="lightbox">', unsafe_allow_html=True)
                st.image(gt_overlay, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_pred:
                st.caption("模型预测 (Prediction)")
                st.markdown('<div class="lightbox">', unsafe_allow_html=True)
                st.image(pred_overlay, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_info:
                accent = "#24a148" if is_mal_pred == is_mal_true else "#da1e28"
                status = "✓ 正确" if correct else "✗ 误判"
                st.markdown(f"""
                <div style="border:1px solid #e0e0e0;border-radius:2px;padding:16px;
                            margin-top:8px;">
                    <p style="font-size:11px;color:#8c8c8c;text-transform:uppercase;
                              letter-spacing:0.32px;margin:0 0 4px;">真实标签</p>
                    <p style="font-size:18px;font-weight:500;color:#161616;margin:0 0 12px;">{true_label}</p>
                    <p style="font-size:11px;color:#8c8c8c;text-transform:uppercase;
                              letter-spacing:0.32px;margin:0 0 4px;">AI 预测</p>
                    <p style="font-size:18px;font-weight:500;color:{'#24a148' if not is_mal_pred else '#da1e28'};margin:0 0 4px;">{pred_label}</p>
                    <p style="font-size:28px;font-weight:300;color:#161616;margin:4px 0;letter-spacing:-0.5px;">{prob*100:.1f}%</p>
                    <p style="font-size:12px;color:{accent};margin:0;">
                        {status}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")


# ============================================================
# 页面一: 单病例智能工作站
# ============================================================
if page == "单病例智能工作站":
    st.markdown("""
    <h2 style="font-weight:300;font-size:32px;letter-spacing:-0.5px;margin:24px 0 4px;">单病例智能工作站</h2>
    <p style="color:#525252;font-size:14px;margin:0 0 24px;">上传单张乳腺超声图像，AI 进行病灶分割与良恶性诊断</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "拖拽或点击上传超声图像",
        type=['png', 'jpg', 'jpeg'],
        help="支持 PNG / JPG 格式，建议 256×256 以上分辨率",
        key="single_uploader",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        with st.spinner(f"{', '.join(active_model_names)} 推理中 (TTA)..."):
            result = run_inference_and_overlay(file_bytes)

        prob = result['prob']
        birads_label = result['birads_label']
        prediction = result['prediction']
        per_model = result['cls_probs_per_model']
        is_mal = prediction == '恶性'
        accent = "#da1e28" if is_mal else "#24a148"
        accent_text = "#da1e28" if is_mal else "#198038"
        status_bg = "#fff1f1" if is_mal else "#f1f8f4"

        # ---- 影像灯箱区 ----
        st.markdown('<div class="animate-in">', unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            st.caption("原始超声影像")
            st.markdown('<div class="lightbox">', unsafe_allow_html=True)
            st.image(result['image_original'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            if inference_preset == "seg_then_classify":
                st.caption("AI 智能解析图 (V10 分割管线)")
            else:
                st.caption("AI 智能解析图")
            st.markdown('<div class="lightbox">', unsafe_allow_html=True)
            st.image(result['overlay'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if is_mal:
                st.caption("红色区域 = AI 检测到的可疑恶性病灶")
            else:
                st.caption("绿色区域 = AI 检测到的良性病灶")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ---- 诊断卡片 ----
        st.markdown('<div class="animate-in">', unsafe_allow_html=True)
        card_cols = st.columns(3)

        with card_cols[0]:
            st.markdown(f"""
            <div style="border:1px solid #e0e0e0;border-radius:2px;padding:24px 20px;
                        border-left:4px solid {accent};background:{status_bg};">
                <p style="font-size:11px;color:#8c8c8c;text-transform:uppercase;
                          letter-spacing:0.32px;margin:0 0 10px;">AI 诊断结论</p>
                <p style="font-size:24px;font-weight:600;color:{accent_text};margin:0;">{prediction}</p>
                <p style="font-size:13px;color:#8c8c8c;margin:4px 0 0;">阈值 = {threshold:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with card_cols[1]:
            st.markdown(f"""
            <div style="border:1px solid #e0e0e0;border-radius:2px;padding:24px 20px;">
                <p style="font-size:11px;color:#8c8c8c;text-transform:uppercase;
                          letter-spacing:0.32px;margin:0 0 10px;">恶性概率</p>
                <p style="font-size:36px;font-weight:300;color:{accent_text};
                          letter-spacing:-0.5px;margin:0;line-height:1.1;">{prob*100:.1f}%</p>
                <div style="height:6px;background:#e0e0e0;margin-top:10px;overflow:hidden;">
                    <div style="height:100%;width:{prob*100:.0f}%;background:{accent};transition:width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(
                min(prob, 1.0),
                text=f"良性 {(1-prob)*100:.0f}%  ←  →  恶性 {prob*100:.0f}%"
            )

        with card_cols[2]:
            st.markdown(f"""
            <div style="border:1px solid #e0e0e0;border-radius:2px;padding:24px 20px;">
                <p style="font-size:11px;color:#8c8c8c;text-transform:uppercase;
                          letter-spacing:0.32px;margin:0 0 10px;">BI-RADS 评级</p>
                <p style="font-size:20px;font-weight:500;color:#161616;margin:0;">{birads_label}</p>
                <p style="font-size:12px;color:#8c8c8c;margin:4px 0 0;">V5.1 BI-RADS 引擎</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ---- V10 融合详情 ----
        if inference_preset == "seg_then_classify":
            st.markdown("""
            <p style="font-size:13px;font-weight:500;color:#525252;text-transform:uppercase;
                      letter-spacing:0.32px;margin-bottom:12px;">V10 分割→分类 融合详情</p>
            """, unsafe_allow_html=True)
            prob_orig = result.get('prob_original', prob)
            prob_roi = result.get('prob_roi', prob)
            v10_cols = st.columns(3)
            with v10_cols[0]:
                st.metric(
                    label="全图 V9 概率",
                    value=f"{prob_orig*100:.1f}%",
                    delta="原始图像分类" if prob_orig < 0.5 else "原始图像分类",
                )
            with v10_cols[1]:
                st.metric(
                    label="ROI V9 概率",
                    value=f"{prob_roi*100:.1f}%",
                    delta="病灶区域分类" if prob_roi < 0.5 else "病灶区域分类",
                )
            with v10_cols[2]:
                delta_text = "融合=恶性" if prob >= 0.5 else "融合=良性"
                st.metric(
                    label="0.5:0.5 融合概率",
                    value=f"{prob*100:.1f}%",
                    delta=delta_text,
                )
            st.markdown("---")

        # ---- 各模型详细输出 ----
        st.markdown("""
        <p style="font-size:13px;font-weight:500;color:#525252;text-transform:uppercase;
                  letter-spacing:0.32px;margin-bottom:12px;">各模型详细输出</p>
        """, unsafe_allow_html=True)
        metric_cols = st.columns(len(per_model))
        for i, (name, p) in enumerate(per_model.items()):
            with metric_cols[i]:
                label = MODEL_REGISTRY.get(name, {}).get('label', name)
                st.metric(
                    label=label,
                    value=f"{p*100:.1f}%",
                    delta="恶性倾向" if p >= 0.5 else "良性倾向",
                )

        st.markdown("---")

        # ---- 生成报告 ----
        st.markdown("""
        <p style="font-size:13px;font-weight:500;color:#525252;text-transform:uppercase;
                  letter-spacing:0.32px;margin-bottom:12px;">结构化电子报告</p>
        """, unsafe_allow_html=True)

        if st.button("生成报告", type="primary", use_container_width=True):
            with st.spinner("正在生成报告..."):
                engine_label = "V10 分割→分类 联合推理" if inference_preset == "seg_then_classify" else "V9 加权 Logit 融合"
                report_html = generate_report_html(
                    image_rgb=result['image_original'],
                    overlay_rgb=result['overlay'],
                    prob=prob,
                    birads_label=birads_label,
                    prediction=prediction,
                    threshold=threshold,
                    engine_label=engine_label,
                )
                pdf_bytes = generate_report_pdf(report_html)

                # PDF 内嵌预览
                pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_data_uri = f"data:application/pdf;base64,{pdf_b64}"

                st.markdown(f"""
                <div style="border:1px solid #e0e0e0;border-radius:2px;overflow:hidden;margin-bottom:16px;">
                    <div style="background:#f4f4f4;padding:8px 16px;border-bottom:1px solid #e0e0e0;
                                display:flex;align-items:center;justify-content:space-between;">
                        <span style="font-size:12px;font-weight:500;color:#161616;">报告预览</span>
                        <span style="font-size:11px;color:#8c8c8c;">PDF · 可翻页 · 可缩放</span>
                    </div>
                    <iframe src="{pdf_data_uri}" width="100%" height="640px"
                            style="border:none;display:block;"></iframe>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label="下载 PDF 报告",
                    data=pdf_bytes,
                    file_name=f"乳腺超声诊断报告_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


# ============================================================
# 页面二: 高通量批量筛查中心
# ============================================================
elif page == "高通量批量筛查中心":
    st.markdown("""
    <h2 style="font-weight:300;font-size:32px;letter-spacing:-0.5px;margin:24px 0 4px;">高通量批量筛查中心</h2>
    <p style="color:#525252;font-size:14px;margin:0 0 24px;">批量上传多张超声图像，AI 调度当前推理模式进行快速筛查并生成报表</p>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "拖拽或点击上传多张超声图像（支持批量选择）",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="可一次性选择多张图片，或拖入整个文件夹",
        key="batch_uploader",
    )

    if uploaded_files:
        st.info(f"已上传 {len(uploaded_files)} 张图像。点击下方按钮开始筛查。")

        if st.button("开始批量筛查", type="primary", use_container_width=True):
            results_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            total = len(uploaded_files)
            for i, file in enumerate(uploaded_files):
                mode_label = "V10 分割→分类" if inference_preset == "seg_then_classify" else ', '.join(active_model_names)
                status_text.markdown(
                    f"正在调度 {mode_label} 处理: **{i+1}/{total}** — {file.name}"
                )

                file_bytes = file.read()
                result = run_inference_and_overlay(file_bytes)

                is_mal = result['prediction'] == '恶性'
                results_list.append({
                    '文件名': file.name,
                    '恶性概率 (%)': f"{result['prob']*100:.1f}%",
                    '恶性概率_raw': result['prob'],
                    'BI-RADS 评级': result['birads_label'],
                    '诊断结论': '恶性' if is_mal else '良性',
                    'is_malignant': is_mal,
                })
                progress_bar.progress((i + 1) / total)

            status_text.markdown(f"筛查完成！共处理 **{total}** 张图像。")
            st.session_state['batch_results'] = results_list

    if 'batch_results' in st.session_state and st.session_state['batch_results']:
        results = st.session_state['batch_results']
        total = len(results)
        n_mal = sum(1 for r in results if r['is_malignant'])

        st.markdown("---")
        st.markdown("""
        <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">筛查结果总览</h3>
        """, unsafe_allow_html=True)

        summ_cols = st.columns(4)
        with summ_cols[0]:
            st.metric("筛查总数", f"{total} 张")
        with summ_cols[1]:
            st.metric("判定恶性", f"{n_mal} 张",
                     delta=f"{n_mal/total*100:.0f}%" if total > 0 else "")
        with summ_cols[2]:
            st.metric("判定良性", f"{total - n_mal} 张",
                     delta=f"{(total-n_mal)/total*100:.0f}%" if total > 0 else "")
        with summ_cols[3]:
            st.metric("当前阈值", f"{threshold:.2f}")

        if total > 0:
            _, center_col, _ = st.columns([1, 2, 1])
            with center_col:
                fig, ax = plt.subplots(figsize=(3, 2))
                labels = ['良性', '恶性']
                sizes = [total - n_mal, n_mal]
                colors = ['#24a148', '#da1e28']
                explode = (0, 0.05) if n_mal > 0 else (0, 0)
                wedges, texts, autotexts = ax.pie(
                    sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize': 13}
                )
                for at in autotexts:
                    at.set_fontweight('bold')
                    at.set_color('#ffffff' if at.get_text().startswith(('0', '1')) else '#161616')
                ax.set_title(f'良恶性分布 (阈值={threshold:.2f})',
                            fontsize=14, fontweight='normal', color='#161616')
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

        df = pd.DataFrame(results)
        df_display = df[['文件名', '恶性概率 (%)', 'BI-RADS 评级', '诊断结论']]

        def highlight_malignant(row):
            if '恶性' in row['诊断结论']:
                return ['background-color: #fff1f1; color: #da1e28; font-weight: 500'] * len(row)
            return [''] * len(row)

        styled_df = df_display.style.apply(highlight_malignant, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)

        csv_data = df_display.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="导出筛查报表 (CSV)",
            data=csv_data,
            file_name=f"乳腺超声筛查报表_阈值{threshold:.2f}_{time.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# 页面三: 算法技术白皮书
# ============================================================
elif page == "算法技术白皮书":
    st.markdown("""
    <h2 style="font-weight:300;font-size:32px;letter-spacing:-0.5px;margin:24px 0 4px;">算法技术白皮书</h2>
    <p style="color:#525252;font-size:14px;margin:0 0 24px;">深度解析系统背后的核心技术架构与实验迭代历程</p>
    """, unsafe_allow_html=True)

    # ── 1. 数据集洞察 ──
    st.markdown("---")
    st.markdown("""
    <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">1. 数据集洞察</h3>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#525252;">通过对 1,875 张乳腺超声图像的探索性分析，团队识别出以下关键数据特征：</p>
    """, unsafe_allow_html=True)

    eda_img_path = os.path.join(_PROJECT_ROOT, "docs", "eda_report.png")
    if not os.path.exists(eda_img_path):
        eda_img_path = os.path.join(_PROJECT_ROOT, "eda_report.png")
    if os.path.exists(eda_img_path):
        _, eda_col, _ = st.columns([1, 4, 1])
        with eda_col:
            st.image(eda_img_path, caption="训练集 EDA 分析报告 (6 个子图)", use_container_width=True)
        st.caption("上图包含: (a)良恶性分布 (b)病理组织学类型 (c)图像尺寸散点 (d)BBOX面积分布 (e)掩膜面积分布 (f)采集设备分布 — 训练集1,875张, 良性:恶性≈2.09:1, 图像宽度240-579px, 测试集平均宽度为训练集的1.9倍, 恶性病灶面积约为良性的2倍")
    else:
        st.warning("EDA 图表未找到，请确保 eda_report.png 存在于 docs/ 或项目根目录。")

    eda_cols = st.columns(3)
    with eda_cols[0]:
        st.markdown("""
        <div class="card card-fill">
        <p style="font-weight:500;font-size:14px;margin:0 0 8px;">尺寸分布偏移</p>
        <p style="font-size:13px;color:#525252;margin:0;line-height:1.6;">
        图像从 200×200 到 600×600+ 不等；固定 Resize(256,256) 为最优预处理；RandomResizedCrop 破坏纵横比 → AUC 下降
        </p>
        </div>
        """, unsafe_allow_html=True)
    with eda_cols[1]:
        st.markdown("""
        <div class="card card-fill">
        <p style="font-weight:500;font-size:14px;margin:0 0 8px;">类别不平衡</p>
        <p style="font-size:13px;color:#525252;margin:0;line-height:1.6;">
        恶性样本约占 35%；FocalLoss(α=0.7, γ=2) 有效缓解；不需要额外上采样/下采样
        </p>
        </div>
        """, unsafe_allow_html=True)
    with eda_cols[2]:
        st.markdown("""
        <div class="card card-fill">
        <p style="font-weight:500;font-size:14px;margin:0 0 8px;">患者级数据泄漏</p>
        <p style="font-size:13px;color:#525252;margin:0;line-height:1.6;">
        811/1064 病例有配对左右乳图像；必须按 Case ID 划分训练/验证集；违反此规则 AUC 虚高 0.02+
        </p>
        </div>
        """, unsafe_allow_html=True)

    # ── 2. 算法演进路线 ──
    st.markdown("---")
    st.markdown("""
    <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 8px;">2. 算法演进路线</h3>
    <p style="color:#525252;font-size:13px;margin:0 0 20px;">从基线到分割→分类联合推理的完整探索历程 — 35 个实验，10 个版本迭代</p>
    """, unsafe_allow_html=True)

    timeline_items = [
        ("V1: Baseline 建立",
         "ResNet34 + FocalLoss + 样本级划分 · AUC 0.905 (数据泄漏警告)"),
        ("V2.3: 诚实基线",
         "患者级划分 + 固定 Resize 256 + CosineAnnealing · AUC 0.903 · 冠军单模型基线"),
        ("V3: 辅助任务探索 (9 实验)",
         "BI-RADS / Histology / BBOX / Mask-Guided / 5-Fold · 全部未能超越 V2.3"),
        ("V4: 架构升级尝试 (5 实验)",
         "UNet++ / DenseNet121 / EfficientNet-B1 · 全部过拟合 · 更强 ≠ 更好"),
        ("V5: 异构架构融合",
         "3 模型: UNet + U-Net+BI-RADS + UNet++ · AUC 0.9073 · 首次突破 0.903 墙"),
        ("V6-V7.4: 多尺度探索",
         "LoMix 组合损失 / 4-scale BCE · 15 个分割损失项 → 严重过拟合 (gap 0.06)"),
        ("V7: 异构融合历史记录",
         "4 模型 (V2.3+V5.1+V5.2+SAM) · 原记录 AUC 0.9093"),
        ("V9: 稀疏加权 logit 融合",
         "DISTILL+SOUP+V2.3 加权 · AUC 0.9139 · best F1=0.7981@0.545 · 纯分类最佳"),
        ("🏆 V10: 分割→分类联合推理 (当前最佳)",
         "MiT-B2 UNet 分割病灶 + V9 全图/ROI 双路分类 · AUC 0.9225 · F1=0.7971@0.55 · 外部 AUC 0.9232"),
    ]

    for title, desc in timeline_items:
        st.markdown(f"""
        <div class="timeline-entry">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── 3. 性能对比 ──
    st.markdown("---")
    st.markdown("""
    <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">3. 性能对比</h3>
    <p style="color:#8c8c8c;font-size:12px;">以 test_ensemble.py --fusion prob_mean 输出为准</p>
    """, unsafe_allow_html=True)

    metrics = ['AUC', 'Recall', 'Specificity', 'F1 Score']
    v23 = [0.9033, 0.833, 0.828, 0.761]
    fusion = [0.9093, 0.886, 0.808, 0.775]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    x = np.arange(len(metrics))
    width = 0.35
    axes[0].bar(x - width/2, v23, width, label='V2.3 单模型',
                color='#4589ff', edgecolor='white', linewidth=0.5)
    axes[0].bar(x + width/2, fusion, width, label='4 模型融合(历史)',
                color='#8c8c8c', edgecolor='white', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=12)
    axes[0].set_ylim(0.7, 0.95)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('V2.3 vs 4-Model Ensemble', fontsize=14, fontweight='normal', color='#161616')
    axes[0].legend(fontsize=11, frameon=False)
    axes[0].grid(axis='y', alpha=0.3, color='#e0e0e0')
    for i in range(4):
        axes[0].text(i - width/2, v23[i] + 0.005, f'{v23[i]:.3f}',
                     ha='center', fontsize=9, fontweight='500')
        axes[0].text(i + width/2, fusion[i] + 0.005, f'{fusion[i]:.3f}',
                     ha='center', fontsize=9, fontweight='500')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Scatter evolution
    experiments = list(range(1, 33))
    auc_values = [
        0.9054, 0.8568, 0.9033, 0.8667, 0.8773, 0.8529, 0.8252, 0.8468,
        0.8869, 0.8877, 0.8992, 0.8915, 0.8932, 0.8746, 0.8867, 0.8713,
        0.8854, 0.8718, 0.8926, 0.8932, 0.9073, 0.8723, 0.8802, 0.8850,
        0.8983, 0.8845, 0.8918, 0.9093, 0.9088, 0.9095, 0.9139, 0.9225,
    ]
    n_hist = 30  # V1-V8 历史实验
    n_new = 2    # V9, V10
    auc_values = auc_values[:n_hist + n_new]

    colors_line = []
    for i, v in enumerate(auc_values):
        if i >= n_hist:
            colors_line.append('#24a148')  # V9/V10 green
        elif v >= 0.907:
            colors_line.append('#da1e28')
        elif v >= 0.90:
            colors_line.append('#4589ff')
        else:
            colors_line.append('#8c8c8c')
    axes[1].scatter(experiments[:n_hist], auc_values[:n_hist], c=colors_line[:n_hist], s=40, zorder=5)
    axes[1].scatter(experiments[n_hist:], auc_values[n_hist:], c=colors_line[n_hist:], s=70, zorder=6, marker='D', edgecolors='#161616', linewidth=0.5)
    axes[1].plot(experiments, auc_values, color='#e0e0e0', alpha=0.8, zorder=3)
    axes[1].axhline(y=0.9033, color='#4589ff', linestyle='--', alpha=0.4, label='V2.3 baseline')
    axes[1].axhline(y=0.9225, color='#24a148', linestyle='--', alpha=0.4, label='V10 (current best)')
    axes[1].set_xlabel('Experiment #', fontsize=12)
    axes[1].set_ylabel('Test AUC', fontsize=12)
    axes[1].set_title('33 Experiments: Test AUC Evolution', fontsize=14, fontweight='normal', color='#161616')
    axes[1].legend(fontsize=10, frameon=False)
    axes[1].grid(alpha=0.3, color='#e0e0e0')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.caption("核心发现: 历史突破来自异构融合的错误模式多样性；V10 证明解耦分割→分类是有效的泛化提升策略（外部数据 +1.23% AUC）。")

    # ── 4. 核心技术栈 ──
    st.markdown("---")
    st.markdown("""
    <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">4. 核心技术栈</h3>
    """, unsafe_allow_html=True)

    tech_data = pd.DataFrame([
        ["骨干网络", "ResNet34 (ImageNet 预训练)", "小数据集上最优; 更大架构均过拟合"],
        ["分割架构", "U-Net / UNet++", "多任务学习辅助分类"],
        ["分类损失", "FocalLoss (α=0.7, γ=2)", "6 种损失函数对比后的最优选择"],
        ["分割损失", "BCEWithLogitsLoss", "Dice/Tversky 在 1507 样本上均更差"],
        ["学习率调度", "CosineAnnealingLR (Tmax=35)", "比 ReduceLROnPlateau 更稳定"],
        ["混合精度", "AMP (torch.amp.autocast)", "训练加速 1.5×, 无精度损失"],
        ["数据增强", "HFlip + Affine + GaussNoise", "固定 Resize, 无随机裁剪"],
        ["TTA", "水平翻转 (仅1种)", "多尺度 TTA 有害"],
        ["优化器", "AdamW + SAM (Sharpness-Aware)", "SAM 提供独特错误模式用于融合"],
        ["分割管线 (V10)", "MiT-B2 UNet 粗分割×2 + ROI 精修×1", "独立分割定位病灶，分类解耦泛化更好"],
        ["融合策略 (V9)", "prob_mean + compact 蒸馏/权重汤", "少模型压缩异构互补信息"],
        ["分割→分类 (V10)", "0.5×全图V9 + 0.5×ROI V9", "零训练成本，域外泛化收益 > 域内"],
    ], columns=["组件", "选型", "决策理由"])

    st.dataframe(tech_data, use_container_width=True, hide_index=True)

    # ── 5. 核心经验 ──
    st.markdown("---")
    st.markdown("""
    <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">5. 35 实验核心经验</h3>
    """, unsafe_allow_html=True)

    insights_data = [
        ("pos", "✓", "患者级划分", "避免 0.02+ AUC 虚高。同一患者的多张图像必须在同一个 split。"),
        ("pos", "✓", "异质模型融合", "唯一突破单模型天花板的方法。融合增益来自错误模式多样性。"),
        ("pos", "✓", "固定 Resize 256", "RandomResizedCrop 破坏纵横比, 导致 AUC 下降 0.026。"),
        ("pos", "✓", "FocalLoss(α=0.7,γ=2)", "6 种分类损失对比的最优解, 有效应对 35%/65% 不平衡。"),
        ("pos", "✓", "概率平均 > Stacking", "Stacking 在 368 样本验证集上过拟合; 简单平均更稳健。"),
        ("neg", "✗", "多尺度架构/损失", "LoMix 15 个分割损失项和 4-scale BCE 均严重过拟合。"),
        ("neg", "✗", "更强 Encoder", "DenseNet121, EfficientNet-B1 全部过拟合。1507 样本不足以训练大模型。"),
        ("neg", "✗", "Dice/Tversky 损失", "在 1507 样本上降低 AUC。BCEForLogitsLoss 更稳定。"),
        ("neg", "✗", "多尺度 TTA", "非训练尺度推理有害。仅水平翻转 TTA 有效。"),
        ("neg", "✗", "SWA / EMA", "小数据集无增益, 最佳单 checkpoint 优于 SWA 平均。"),
        ("pos", "✓", "解耦分割→分类 (V10)", "独立分割模型+已有分类器组合，零训练成本，外部数据 Sens +4.4%。"),
        ("pos", "✓", "ROI 聚焦抗域偏移", "粗糙分割也能辅助分类；外部 Dice 仅 0.48 但 AUC 仍 +1.23%。"),
    ]

    for tag_class, icon, title, desc in insights_data:
        st.markdown(f"""
        <div class="insight-item">
            <span class="tag {tag_class}">{icon}</span>
            <div>
                <span style="font-weight:500;">{title}</span>
                <span class="text"> — {desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 6. 完整实验排名表 ──
    st.markdown("---")
    st.markdown("""
    <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">6. 完整实验排名</h3>
    <p style="color:#8c8c8c;font-size:12px;margin:0 0 16px;">35 实验按 Test AUC 降序</p>
    """, unsafe_allow_html=True)

    ranking_data = [
        ["🏆 V10 分割→分类 (V9+MiT-B2 seg)",  "0.9225", "—",    "0.776@0.55",  "0.797", "0.5原图+0.5ROI融合", "当前最佳"],
        ["V9 稀疏加权 (SOUP+DISTILL+V2.3)",  "0.9139", "—",    "0.810@0.545", "0.798", "logit 加权融合", "纯分类最佳"],
        ["7模型融合 (V5+SAM+3Seed)",           "0.9095", "—",    "0.833@0.40",  "0.773", "异质+种子(旧公式)", "历史记录"],
        ["4模型融合 (V2.3+V5+SAM)",            "0.9093", "—",    "0.853@0.45",  "0.772", "异质+SAM(旧公式)", "历史记录"],
        ["6模型融合 (V5+3Seed)",               "0.9088", "—",    "0.795@0.45",  "0.770", "异质+种子(旧公式)", ""],
        ["SOUP 权重汤单模型",                  "0.9083", "—",    "0.781@0.55",  "0.787", "Greedy Soup", "V9 主排序器"],
        ["DISTILL 蒸馏单模型",                 "0.9074", "—",    "0.848@0.55",  "0.777", "4教师→1学生", "近4模型融合"],
        ["3模型融合 (V2.3+V5.1+V5.2)",        "0.9073", "—",    "0.876@0.45",  "0.770", "异质架构融合", "首破 0.903 墙"],
        ["2模型融合 (V2.3+V5.1)",             "0.9057", "—",    "0.867@0.45",  "0.775", "两模型融合", ""],
        ["V1 (样本级划分)",                    "0.9054", "0.9588","0.805@0.55",  "0.763", "数据泄露警告", ""],
        ["V2.3 单模型",                       "0.9033", "0.9242","0.848@0.40",  "0.764", "冠军单模型基线", "22 实验未超越"],
        ["V3.6 BI-RADS 辅助头",               "0.8992", "0.9358","0.833@0.50",  "0.787", "单模型 F1 最高", ""],
        ["V7.5 S456",                         "0.8983", "0.9112","0.833@0.45",  "0.748", "Seed=456", ""],
        ["V3.9 5-fold 融合",                  "0.8932", "0.9277","0.891@0.55",  "0.702", "同质融合(负增益)", ""],
        ["V5.2 UNet++ SWA",                   "0.8932", "0.9260","0.876@0.45",  "0.726", "SWA 无效", ""],
        ["V5.1 BI-RADS SWA",                  "0.8926", "0.9290","0.867@0.45",  "0.728", "SWA 无效", ""],
        ["V7.6 SAM",                           "0.8918", "0.9310","0.829@0.55",  "0.757", "SAM 单模型", "融合贡献大"],
        ["V3.7 EMA+SE-ResNet50",               "0.8915", "0.9295","0.905@0.50",  "0.653", "Rec 极高/Spec 崩溃", ""],
        ["V3.5 Histology 辅助头",              "0.8877", "0.9298","0.781@0.50",  "0.768", "", ""],
        ["V2.1 384×384",                       "0.8876", "0.9419","0.857@0.40",  "0.745", "分辨率过高", ""],
        ["V3.4 BBOX 辅助头",                   "0.8869", "0.9348","0.852@0.50",  "0.743", "", ""],
        ["V4.2 UNet++",                        "0.8867", "0.9395","0.824@0.50",  "0.766", "严重过拟合", ""],
        ["V4.4 DenseNet121",                   "0.8854", "0.9098","0.805@0.50",  "0.751", "更大架构过拟合", ""],
        ["V7.5 S123",                          "0.8850", "0.9479","0.800@0.45",  "0.712", "Seed=123", ""],
        ["V7.5 S789",                          "0.8845", "0.9305","0.800@0.45",  "0.727", "Seed=789", ""],
        ["V7.4 简化多尺度",                    "0.8802", "0.9322","0.819@0.45",  "0.694", "4-scale BCE", "过拟合"],
        ["V2.5 多尺度裁剪",                    "0.8773", "0.9221","0.871@0.40",  "0.754", "RRC 破坏纵横比", ""],
        ["V8.2 ROI 双分支",                    "0.8756", "0.8956","0.786@0.55",  "0.722", "修正后评测", "未超 compact"],
        ["V8.3 ROI+shape 特征",                "0.8752", "0.9018","0.752@0.55",  "0.728", "修正后评测", "Val 虚高"],
        ["V4.1 BI-RADS+Tversky",              "0.8746", "0.9253","0.857@0.50",  "0.726", "", ""],
        ["V6 LoMix",                           "0.8723", "0.9330","0.829@0.40",  "0.691", "15 个损失项", "gap 0.061"],
        ["V4.5 EfficientNet-B1",              "0.8718", "0.8998","0.833@0.50",  "0.714", "更强架构过拟合", ""],
        ["V4.3 UNet+++BI-RADS+Tversky",       "0.8713", "0.9145","0.786@0.50",  "0.727", "", ""],
        ["V8.1 纯分类全图",                    "0.8692", "0.8929","0.752@0.55",  "0.761", "去掉 decoder", "远低于 compact"],
        ["V2.4 BI-RADS 样本权重",               "0.8667", "0.9282","0.857@0.40",  "0.707", "严重过拟合", ""],
        ["V2 全改",                            "0.8568", "0.9418","—",           "—",    "全改(崩溃)", ""],
        ["V3.1 两阶段冻结",                    "0.8529", "0.8937","0.910@0.50",  "0.671", "", ""],
        ["V3.3 BBOX ROI",                      "0.8468", "0.9050","0.795@0.50",  "0.684", "", ""],
        ["V3.2 Mask-Guided",                   "0.8252", "0.8976","0.881@0.50",  "0.622", "", ""],
    ]

    df_ranking = pd.DataFrame(ranking_data,
        columns=["版本", "Test AUC", "Val AUC", "Recall@best", "F1", "分类/备注", "说明"])

    # 根据当前推理模式高亮对应实验行
    preset_to_experiment = {
        "seg_then_classify": "V10 分割→分类",
        "compact": "V9 稀疏加权",
        "compact_weighted": "V9 稀疏加权",
        "ensemble4": "4模型融合 (V2.3+V5+SAM)",
        "distill": "DISTILL",
        "soup": "SOUP",
        "v23": "V2.3 单模型",
    }
    highlight_name = preset_to_experiment.get(inference_preset, "")

    def highlight_row(row):
        if highlight_name and highlight_name in row["版本"]:
            return ['background-color: #e8f1ff; font-weight: 500'] * len(row)
        return [''] * len(row)

    styled_ranking = df_ranking.style.apply(highlight_row, axis=1)
    st.dataframe(styled_ranking, use_container_width=True, hide_index=True,
                 column_config={
                     "Test AUC": st.column_config.NumberColumn(format="%.4f"),
                     "Val AUC": None,
                 })

    # ── 7. V9 阈值扫描 ──
    st.markdown("---")
    st.markdown("""
    <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">7. V9 阈值扫描 (纯分类参考)</h3>
    """, unsafe_allow_html=True)

    col_v9a, col_v9b = st.columns(2)

    with col_v9a:
        st.markdown("""
        <p style="font-weight:500;font-size:14px;margin:0 0 12px;">V9 Weighted Logit 各阈值表现</p>
        """, unsafe_allow_html=True)
        v9_thresh_data = [
            ["0.40", "0.6368", "0.9476", "0.4874", "0.4704", "0.6288"],
            ["0.45", "0.7249", "0.9190", "0.6316", "0.5452", "0.6844"],
            ["0.50", "0.8099", "0.8714", "0.7803", "0.6559", "0.7485"],
            ["0.545 (best F1)", "—", "0.8095", "0.8947", "—", "0.7981"],
            ["0.55", "0.8671", "0.7952", "0.9016", "0.7952", "0.7952"],
        ]
        df_v9t = pd.DataFrame(v9_thresh_data,
            columns=["阈值", "Acc", "Recall", "Specificity", "Precision", "F1"])
        st.dataframe(df_v9t, use_container_width=True, hide_index=True)

    with col_v9b:
        st.markdown("""
        <p style="font-weight:500;font-size:14px;margin:0 0 12px;">Recall / Spec / F1 vs Threshold</p>
        """, unsafe_allow_html=True)
        fig_thresh, ax_thresh = plt.subplots(figsize=(5, 3.5))
        thresh_vals = [0.40, 0.45, 0.50, 0.55]
        rec_vals  = [0.9476, 0.9190, 0.8714, 0.7952]
        spec_vals = [0.4874, 0.6316, 0.7803, 0.9016]
        f1_vals   = [0.6288, 0.6844, 0.7485, 0.7952]
        ax_thresh.plot(thresh_vals, rec_vals, 'o-', color='#4589ff', linewidth=2, label='Recall')
        ax_thresh.plot(thresh_vals, spec_vals, 's-', color='#24a148', linewidth=2, label='Specificity')
        ax_thresh.plot(thresh_vals, f1_vals, 'D-', color='#da1e28', linewidth=2, label='F1')
        ax_thresh.set_xlabel('Threshold', fontsize=11)
        ax_thresh.set_ylabel('Score', fontsize=11)
        ax_thresh.set_ylim(0.4, 1.0)
        ax_thresh.legend(fontsize=10, frameon=False)
        ax_thresh.grid(alpha=0.3, color='#e0e0e0')
        ax_thresh.spines['top'].set_visible(False)
        ax_thresh.spines['right'].set_visible(False)
        st.pyplot(fig_thresh, use_container_width=False)
        plt.close(fig_thresh)

    # ── 8. 单模型 vs 融合对比 ──
    st.markdown("---")
    st.markdown("""
    <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">8. 单模型 vs 融合方案对比</h3>
    """, unsafe_allow_html=True)

    fig_compare, ax_comp = plt.subplots(figsize=(6, 2.85))
    models_labels = ['V2.3\n单模型', 'DISTILL\n单模型', 'SOUP\n单模型',
                     'DISTILL\n+SOUP', 'V9 三模型\n加权', 'V10 分割\n→分类']
    models_auc = [0.9033, 0.9074, 0.9083, 0.9114, 0.9139, 0.9225]
    models_f1 = [0.764, 0.7773, 0.7866, 0.7832, 0.7981, 0.7971]
    bar_colors = ['#8c8c8c', '#8c8c8c', '#8c8c8c', '#4589ff', '#da1e28', '#24a148']

    x_pos = np.arange(len(models_labels))
    width = 0.35
    bars_auc = ax_comp.bar(x_pos - width/2, models_auc, width, label='AUC',
                           color=bar_colors, edgecolor='white', linewidth=0.5)
    bars_f1 = ax_comp.bar(x_pos + width/2, models_f1, width, label='F1 @0.55',
                          color=[c + '88' for c in bar_colors], edgecolor='white', linewidth=0.5,
                          hatch='//')
    for bar, val in zip(bars_auc, models_auc):
        ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', fontsize=8, fontweight='500', color='#161616')
    for bar, val in zip(bars_f1, models_f1):
        ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', fontsize=8, fontweight='500', color='#525252')
    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels(models_labels, fontsize=10)
    ax_comp.set_ylim(0.75, 0.94)
    ax_comp.set_ylabel('Score', fontsize=11)
    ax_comp.legend(fontsize=10, frameon=False, loc='lower right')
    ax_comp.grid(axis='y', alpha=0.3, color='#e0e0e0')
    ax_comp.spines['top'].set_visible(False)
    ax_comp.spines['right'].set_visible(False)
    ax_comp.set_title('AUC & F1: Single Models → Ensemble', fontsize=13,
                      fontweight='normal', color='#161616')
    _, center_col, _ = st.columns([1, 3, 1])
    with center_col:
        st.pyplot(fig_compare, use_container_width=True)
    plt.close(fig_compare)

    # ── 9. Val-Test Gap 分析 ──
    st.markdown("---")
    st.markdown("""
    <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">9. Val-Test Gap 分析</h3>
    <p style="color:#525252;font-size:13px;margin:0 0 16px;">每个点代表一个实验，对角线以上 = 过拟合（Val > Test），红色标记 gap > 0.05 的严重过拟合</p>
    """, unsafe_allow_html=True)

    gap_data = [
        ("V1", 0.9588, 0.9054), ("V2", 0.9418, 0.8568), ("V2.1", 0.9419, 0.8876),
        ("V2.3", 0.9242, 0.9033), ("V2.4", 0.9282, 0.8667), ("V2.5", 0.9221, 0.8773),
        ("V3.1", 0.8937, 0.8529), ("V3.2", 0.8976, 0.8252), ("V3.3", 0.9050, 0.8468),
        ("V3.4", 0.9348, 0.8869), ("V3.5", 0.9298, 0.8877), ("V3.6", 0.9358, 0.8992),
        ("V3.7", 0.9295, 0.8915), ("V3.9", 0.9277, 0.8932), ("V4.1", 0.9253, 0.8746),
        ("V4.2", 0.9395, 0.8867), ("V4.3", 0.9145, 0.8713), ("V4.4", 0.9098, 0.8854),
        ("V4.5", 0.8998, 0.8718), ("V5.1", 0.9290, 0.8926), ("V5.2", 0.9260, 0.8932),
        ("V6", 0.9330, 0.8723), ("V7.4", 0.9322, 0.8802), ("V7.5 S456", 0.9112, 0.8983),
        ("V7.5 S123", 0.9479, 0.8850), ("V7.5 S789", 0.9305, 0.8845),
        ("V7.6 SAM", 0.9310, 0.8918), ("V8.1", 0.8929, 0.8692),
        ("V8.2", 0.8956, 0.8756), ("V8.3", 0.9018, 0.8752),
    ]

    fig_gap, ax_gap = plt.subplots(figsize=(4.875, 3.375))
    val_vals = [d[1] for d in gap_data]
    test_vals = [d[2] for d in gap_data]
    gaps = [v - t for v, t in zip(val_vals, test_vals)]
    gap_colors = ['#da1e28' if g > 0.05 else '#4589ff' if g > 0.03 else '#8c8c8c' for g in gaps]
    ax_gap.scatter(val_vals, test_vals, c=gap_colors, s=50, zorder=5, alpha=0.8)
    # 对角参考线 (完美情况 Val = Test)
    lim_min = min(min(val_vals), min(test_vals)) - 0.01
    lim_max = max(max(val_vals), max(test_vals)) + 0.01
    ax_gap.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='#e0e0e0', linewidth=1, zorder=2)
    # 标注 gap > 0.05 的点
    for i, (name, v, t) in enumerate(gap_data):
        if gaps[i] > 0.05:
            ax_gap.annotate(name, (v, t), fontsize=7, color='#da1e28',
                           xytext=(3, 3), textcoords='offset points')
    ax_gap.set_xlabel('Val AUC', fontsize=11)
    ax_gap.set_ylabel('Test AUC', fontsize=11)
    ax_gap.set_title('Val AUC vs Test AUC (Gap Analysis)', fontsize=13,
                     fontweight='normal', color='#161616')
    ax_gap.grid(alpha=0.3, color='#e0e0e0')
    ax_gap.spines['top'].set_visible(False)
    ax_gap.spines['right'].set_visible(False)
    _, center_col, _ = st.columns([1, 3, 1])
    with center_col:
        st.pyplot(fig_gap, use_container_width=True)
    plt.close(fig_gap)
    st.caption("红色点 (gap > 0.05): 严重过拟合 — V2, V2.4, V6 LoMix, V7.4 等 • "
               "蓝色点 (gap 0.03~0.05): 中度过拟合 • 灰色点: 正常泛化")

    # ── 10. V9 权重占比 ──
    st.markdown("---")
    col_w1, col_w2 = st.columns([1, 1])

    with col_w1:
        st.markdown("""
        <h3 style="font-weight:300;font-size:24px;letter-spacing:-0.3px;margin:0 0 16px;">10. V9 融合权重 & V10 推理架构</h3>
        <p style="color:#525252;font-size:13px;margin:0 0 12px;">左: V9 weighted logit 权重 — SOUP 是绝对主排序器 &nbsp;|&nbsp; 右: V10 分割→分类架构</p>
        """, unsafe_allow_html=True)
        fig_pie, ax_pie = plt.subplots(figsize=(2.625, 2.25))
        weights = [0.0434, 0.8786, 0.0780]
        labels = ['DISTILL\n4.3%', 'SOUP\n87.9%', 'V2.3\n7.8%']
        colors_pie = ['#8c8c8c', '#4589ff', '#525252']
        wedges, texts, autotexts = ax_pie.pie(
            weights, labels=labels, colors=colors_pie,
            autopct='', startangle=90,
            textprops={'fontsize': 13},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for t in texts:
            t.set_fontweight('500')
        # 手动标百分比
        for i, (w, wedge) in enumerate(zip(weights, wedges)):
            ang = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        ax_pie.set_title('V9 Weighted Logit Fusion', fontsize=13,
                        fontweight='normal', color='#161616')
        st.pyplot(fig_pie, use_container_width=False)
        plt.close(fig_pie)

    with col_w2:
        st.markdown("""
        <div style="padding-top:20px;">
        <div class="card" style="margin-bottom:12px;">
            <p style="font-weight:500;font-size:14px;margin:0 0 8px;">V9 融合公式</p>
            <p style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#525252;margin:0;line-height:1.6;">
                p = σ(0.0434·logit<sub>D</sub> + 0.8786·logit<sub>S</sub> + 0.0780·logit<sub>V</sub>)
            </p>
        </div>
        <div class="card" style="margin-bottom:12px;">
            <p style="font-weight:500;font-size:14px;margin:0 0 8px;">V10 推理架构</p>
            <p style="font-size:12px;color:#525252;margin:0;line-height:1.6;">
                <b>全图支路:</b> 原图 → V9 → P<sub>orig</sub> (50%)<br>
                <b>ROI 支路:</b> 分割 → ROI(m=1.0) → V9 → P<sub>roi</sub> (50%)<br>
                <b>融合:</b> P = 0.5·P<sub>orig</sub> + 0.5·P<sub>roi</sub>
            </p>
        </div>
        <div class="card">
            <p style="font-weight:500;font-size:14px;margin:0 0 8px;">V10 分割管线</p>
            <p style="font-size:12px;color:#525252;margin:0;line-height:1.6;">
                MiT-B2 UNet 粗分割×2 (平均)<br>
                → ROI 精修×1<br>
                → 0.75×粗 + 0.25×精<br>
                → 阈值 0.38 → 后处理
            </p>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
