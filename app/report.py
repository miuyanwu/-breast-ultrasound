"""
临床报告生成模块
  generate_report_html(): 生成专业医疗报告 HTML (Apple×IBM 融合风格)
  generate_report_pdf():   HTML → PDF 二进制 (weasyprint)
  _image_to_base64():      图像 → base64 嵌入
"""
import base64
import io
import numpy as np
from PIL import Image


def _image_to_base64(image_rgb: np.ndarray) -> str:
    """numpy RGB 图像 → base64 Data URI 字符串"""
    pil_img = Image.fromarray(image_rgb.astype('uint8'), mode='RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def generate_report_html(image_rgb: np.ndarray,
                         overlay_rgb: np.ndarray,
                         prob: float,
                         birads_label: str,
                         prediction: str,
                         threshold: float,
                         engine_label: str = "V9 加权 Logit 融合") -> str:
    """生成结构化电子报告的 HTML 字符串。

    Args:
        image_rgb:   原图 [H,W,3] uint8 RGB
        overlay_rgb: AI 叠加图 [H,W,3] uint8 RGB
        prob:        恶性概率 [0, 1]
        birads_label: BI-RADS 标签字符串
        prediction:  '恶性' 或 '良性'
        threshold:   当前诊断阈值

    Returns:
        完整的 HTML/CSS 字符串
    """
    img_b64 = _image_to_base64(image_rgb)
    overlay_b64 = _image_to_base64(overlay_rgb)

    prob_pct = prob * 100
    is_malignant = prediction == '恶性'

    accent_color = "#da1e28" if is_malignant else "#24a148"
    accent_text = "#da1e28" if is_malignant else "#198038"
    status_bg = "#fff1f1" if is_malignant else "#f1f8f4"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'IBM Plex Sans', 'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', sans-serif;
    font-size: 14px;
    font-weight: 400;
    line-height: 1.6;
    color: #161616;
    background: #ffffff;
    -webkit-font-smoothing: antialiased;
  }}

  .report {{
    max-width: 780px;
    margin: 0 auto;
    padding: 48px 0;
  }}

  /* ── Header ── */
  .report-header {{
    padding: 0 0 24px;
    margin-bottom: 32px;
    border-bottom: 2px solid #4589ff;
  }}
  .report-header h1 {{
    font-family: 'IBM Plex Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    font-size: 28px;
    font-weight: 300;
    letter-spacing: -0.5px;
    color: #161616;
    margin: 0 0 4px;
  }}
  .report-header .sub {{
    font-size: 13px;
    font-weight: 400;
    color: #525252;
    letter-spacing: 0.16px;
  }}

  /* ── Meta grid ── */
  .meta-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border: 1px solid #e0e0e0;
    margin-bottom: 28px;
  }}
  .meta-item {{
    padding: 12px 16px;
    border-right: 1px solid #e0e0e0;
    border-bottom: 1px solid #e0e0e0;
  }}
  .meta-item:nth-child(3n) {{ border-right: none; }}
  .meta-item:nth-last-child(-n+3) {{ border-bottom: none; }}
  .meta-item .label {{
    font-size: 11px;
    font-weight: 400;
    color: #8c8c8c;
    text-transform: uppercase;
    letter-spacing: 0.32px;
    margin-bottom: 4px;
  }}
  .meta-item .value {{
    font-size: 14px;
    font-weight: 500;
    color: #161616;
  }}

  /* ── Images ── */
  .image-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 28px;
  }}
  .image-card {{
    border: 1px solid #e0e0e0;
    border-radius: 2px;
    overflow: hidden;
  }}
  .image-card .img-label {{
    font-size: 12px;
    font-weight: 500;
    color: #525252;
    padding: 10px 14px;
    background: #f4f4f4;
    border-bottom: 1px solid #e0e0e0;
    letter-spacing: 0.16px;
  }}
  .image-card img {{
    width: 100%;
    display: block;
    background: #1d1d1f;
  }}

  /* ── Diagnosis cards ── */
  .diag-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
  }}
  .diag-card {{
    border: 1px solid #e0e0e0;
    border-radius: 2px;
    padding: 20px 18px;
    background: #ffffff;
  }}
  .diag-card.conclusion {{
    border-left: 4px solid {accent_color};
    background: {status_bg};
  }}
  .diag-card .card-title {{
    font-size: 11px;
    font-weight: 400;
    color: #8c8c8c;
    text-transform: uppercase;
    letter-spacing: 0.32px;
    margin-bottom: 10px;
  }}
  .diag-card.conclusion .result-text {{
    font-size: 22px;
    font-weight: 600;
    color: {accent_text};
    letter-spacing: -0.3px;
  }}
  .prob-number {{
    font-size: 32px;
    font-weight: 300;
    color: {accent_text};
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 8px;
  }}
  .prob-bar {{
    height: 6px;
    background: #e0e0e0;
    border-radius: 0;
    overflow: hidden;
    margin-top: 6px;
  }}
  .prob-bar-fill {{
    height: 100%;
    background: #24a148;
    width: {prob_pct:.1f}%;
  }}
  .birads-value {{
    font-size: 18px;
    font-weight: 500;
    color: #161616;
  }}

  /* ── Divider ── */
  .divider {{
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 32px 0;
  }}

  /* ── Footer ── */
  .report-footer {{
    padding: 0;
    font-size: 12px;
    color: #8c8c8c;
    line-height: 2;
  }}
  .report-footer .disclaimer {{
    color: #da1e28;
    font-weight: 500;
  }}
  .report-footer .meta-line {{
    color: #8c8c8c;
  }}

  @media print {{
    body {{ background: #fff; }}
    .report {{ max-width: 100%; padding: 24px 0; }}
  }}
</style>
</head>
<body>
<div class="report">

  <div class="report-header">
    <h1>乳腺超声智能辅助诊断报告</h1>
    <div class="sub">Breast Ultrasound AI-Assisted Diagnosis Report &nbsp;·&nbsp; 生物医学工程竞赛</div>
  </div>

  <div class="meta-grid">
    <div class="meta-item">
      <div class="label">检查编号</div>
      <div class="value">BUS-2026-{abs(hash(str(prob))) % 100000:05d}</div>
    </div>
    <div class="meta-item">
      <div class="label">检查日期</div>
      <div class="value">2026-06-12</div>
    </div>
    <div class="meta-item">
      <div class="label">检查部位</div>
      <div class="value">乳腺 (超声)</div>
    </div>
    <div class="meta-item">
      <div class="label">诊断阈值</div>
      <div class="value">{threshold:.2f}</div>
    </div>
    <div class="meta-item">
      <div class="label">AI 引擎</div>
      <div class="value">{engine_label}</div>
    </div>
    <div class="meta-item">
      <div class="label">参考标准</div>
      <div class="value">AUC 0.9225 / F1 0.7971</div>
    </div>
  </div>

  <div class="image-row">
    <div class="image-card">
      <div class="img-label">原始超声影像</div>
      <img src="{img_b64}" alt="原始超声图">
    </div>
    <div class="image-card">
      <div class="img-label">AI 智能分割定位图</div>
      <img src="{overlay_b64}" alt="AI 分割图">
    </div>
  </div>

  <div class="diag-row">
    <div class="diag-card conclusion">
      <div class="card-title">AI 诊断结论</div>
      <div class="result-text">{prediction}</div>
    </div>
    <div class="diag-card">
      <div class="card-title">恶性概率</div>
      <div class="prob-number">{prob_pct:.1f}%</div>
      <div class="prob-bar"><div class="prob-bar-fill"></div></div>
    </div>
    <div class="diag-card">
      <div class="card-title">BI-RADS 评级</div>
      <div class="birads-value">{birads_label}</div>
    </div>
  </div>

  <hr class="divider">

  <div class="report-footer">
    <p><span class="disclaimer">⚠ 免责声明：</span>本报告由 AI 辅助生成，仅供临床参考，不构成最终诊断依据。所有诊断决策须由具备执业资格的医师复核确认。</p>
    <p class="meta-line">AI 引擎: V10 分割→分类 联合推理 &nbsp;|&nbsp; AUC=0.9225 &nbsp;|&nbsp; 生成时间: 2026-06-12</p>
  </div>

</div>
</body>
</html>"""
    return html


def generate_report_pdf(html: str) -> bytes:
    """将报告 HTML 转换为 PDF 二进制数据。

    Args:
        html: generate_report_html() 的完整 HTML 字符串

    Returns:
        PDF 文件的 bytes，可直接用于下载或 base64 编码
    """
    import os as _os
    import platform as _platform

    # Windows: conda 环境需要 GTK3 DLL 加入 PATH
    if _platform.system() == 'Windows':
        _gtk_bin = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                                 '.venv', 'Library', 'bin')
        if not _os.path.isdir(_gtk_bin):
            for _maybe in [
                _os.path.join(_os.path.dirname(_os.__file__), '..', 'Library', 'bin'),
                _os.path.join(_os.path.dirname(_os.__file__), '..', '..', 'Library', 'bin'),
            ]:
                if _os.path.isdir(_os.path.abspath(_maybe)):
                    _gtk_bin = _os.path.abspath(_maybe)
                    break
        if _os.path.isdir(_gtk_bin) and _gtk_bin not in _os.environ.get('PATH', ''):
            _os.environ['PATH'] = _gtk_bin + _os.pathsep + _os.environ.get('PATH', '')
    # Linux: weasyprint 使用 apt 安装的系统 Pango/Cairo 库，无需 PATH 操作

    from weasyprint import HTML

    pdf_bytes = HTML(string=html).write_pdf()
    return pdf_bytes
