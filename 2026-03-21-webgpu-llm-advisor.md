# WebGPU LLM Stats Advisor — Implementation Plan (Replaces Task 6 of Interpretation Plan)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Plan 6 of 10**

**Goal:** Replace the Anthropic API-powered "Ask About This" component with a fully client-side WebGPU LLM that runs Qwen 2.5 1.5B directly in the user's browser — zero API keys, zero server costs, full privacy.

**Architecture:** A Streamlit custom component embeds an HTML/JS page (via `st.components.v1.html()`) that loads WebLLM from CDN, initializes Qwen2.5-1.5B-Instruct (q4f16_1 quantization, ~1.6GB VRAM), and provides a chat interface for asking follow-up questions about statistics. The Python side passes the statistical context as a JSON-encoded data attribute. The JS side handles model loading, caching (models persist in browser CacheStorage after first download), streaming inference, and a clean chat UI. No build toolchain needed — it's a single HTML string with inline JS.

**Tech Stack:**
- **Inference:** WebLLM (`@mlc-ai/web-llm` via CDN), WebGPU
- **Model:** `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` (~1.6GB VRAM, 4096 context)
- **Integration:** `st.components.v1.html()` (iframe-based, no React build needed)
- **Fallback:** Graceful degradation when WebGPU is unavailable

**Prerequisite:** Interpretation Engine (Plan 5, Tasks 1-5) must be implemented first.

---

## Context: Why WebGPU + In-Browser LLM?

Traditional approach: user clicks "Ask about this" → request goes to Anthropic API → requires API key management, costs money, sends data to external server.

WebGPU approach: user clicks "Ask about this" → Qwen 2.5 1.5B runs directly in their browser tab → no API key, no server, no data leaves the device, works offline after first model download.

The tradeoffs are real:
- **Quality:** Qwen 1.5B is much less capable than Sonnet. It can handle "explain this VIF score" but will struggle with nuanced multi-factor reasoning.
- **First load:** ~1.6GB model download on first use (cached in CacheStorage after that — instant on subsequent visits). Add a download progress indicator showing estimated time remaining based on download speed. Add a retry button for interrupted downloads. Note: WebLLM uses CacheStorage which handles partial caches, but a network interruption during initial download requires restarting.
- **Speed:** ~10-30 tokens/sec on a modern GPU (RTX 3060+), slower on integrated graphics. Your RTX 5070 Ti will handle it extremely well. Users with integrated GPUs or older hardware may experience degraded performance or VRAM limitations.
- **Compatibility:** Requires WebGPU (Chrome 113+, Edge 113+, Firefox behind flag). Safari 18+ supports WebGPU via Metal backend. Safari 17 has experimental support behind a flag. Earlier versions receive the fallback message.

For a stats interpretation use case, the quality tradeoff is acceptable — the questions are structured ("what does this VIF mean?"), the context is provided in full by the system prompt, and the model only needs to synthesize and explain rather than reason from scratch.

---

## Task 1: WebGPU Capability Detection Utility

**Files:**
- Create: `packages/dashboard/src/dashboard/components/webgpu_check.py`
- Test: `tests/dashboard/test_webgpu_check.py`

**Step 1: Write the failing test**

```python
# tests/dashboard/test_webgpu_check.py
import pytest

from dashboard.components.webgpu_check import (
    get_webgpu_check_script,
    get_fallback_message,
)


def test_webgpu_check_script_returns_string():
    script = get_webgpu_check_script()
    assert isinstance(script, str)
    assert "navigator.gpu" in script
    assert "requestAdapter" in script


def test_fallback_message_returns_html():
    html = get_fallback_message()
    assert isinstance(html, str)
    assert "WebGPU" in html
    assert "Chrome" in html or "browser" in html.lower()
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/dashboard/test_webgpu_check.py -v
```

Expected: FAIL.

**Step 3: Implement the utility**

```python
# packages/dashboard/src/dashboard/components/webgpu_check.py
"""WebGPU capability detection utilities.

Provides JavaScript snippets and fallback HTML for checking whether
the user's browser supports WebGPU, which is required for WebLLM.
"""


def get_webgpu_check_script() -> str:
    """Return JS that checks WebGPU availability and sets a global flag."""
    return """
    async function checkWebGPU() {
        if (!navigator.gpu) {
            return { available: false, reason: 'WebGPU API not found in this browser.' };
        }
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                return { available: false, reason: 'No WebGPU adapter found. Your GPU may not be supported.' };
            }
            const info = await adapter.requestAdapterInfo();
            const maxBuf = adapter.limits.maxBufferSize;
            const vramWarning = (maxBuf < 2 * 1024 * 1024 * 1024)
                ? 'Your GPU may not have enough memory for the LLM model. The advisor may fail to load.'
                : null;
            return {
                available: true,
                gpu: info.description || info.device || 'Unknown GPU',
                vendor: info.vendor || 'Unknown',
                vramWarning: vramWarning,
            };
        } catch (e) {
            return { available: false, reason: e.message };
        }
    }
    """


In addition to checking `navigator.gpu`, query `adapter.limits.maxBufferSize` to estimate available VRAM. If below 2GB, show a warning: "Your GPU may not have enough memory for the LLM model. The advisor may fail to load."


def get_fallback_message() -> str:
    """Return HTML shown when WebGPU is not available."""
    return """
    <div style="padding: 16px; border: 1px solid #f0ad4e; border-radius: 8px;
                background: #fcf8e3; color: #8a6d3b; font-family: sans-serif;">
        <strong>⚠️ WebGPU not available</strong>
        <p style="margin: 8px 0 0 0; font-size: 14px;">
            The AI stats advisor requires WebGPU, which is available in:
        </p>
        <ul style="margin: 4px 0; font-size: 14px;">
            <li>Google Chrome 113+ (recommended)</li>
            <li>Microsoft Edge 113+</li>
            <li>Firefox Nightly (behind flag)</li>
        </ul>
        <p style="margin: 4px 0 0 0; font-size: 13px; color: #666;">
            The static interpretations above still provide explanations for all statistics.
        </p>
    </div>
    """
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/dashboard/test_webgpu_check.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add WebGPU capability detection utility"
```

---

## Task 2: WebLLM Chat Component — Core HTML/JS

This is the main component. It's a self-contained HTML page with inline JavaScript that:
1. Checks WebGPU availability
2. Loads WebLLM from CDN
3. Downloads/caches the Qwen 2.5 1.5B model
4. Provides a chat UI for asking questions about statistics
5. Streams responses token-by-token

**Files:**
- Create: `packages/dashboard/src/dashboard/components/llm_advisor.py`

**Step 1: Implement the component**

```python
# packages/dashboard/src/dashboard/components/llm_advisor.py
"""In-browser LLM stats advisor powered by WebGPU + WebLLM.

Renders a chat interface inside a Streamlit component where the user
can ask follow-up questions about statistical results. The LLM
(Qwen2.5-1.5B-Instruct) runs entirely in the browser via WebGPU —
no API keys, no server, no data leaves the device.

Usage:
    from dashboard.components.llm_advisor import render_llm_advisor

    render_llm_advisor(
        context="NFL 4th-down model with 17 features...",
        stat_description="VIF for score_differential is 7.3",
    )
"""

import json
import html as html_module
import streamlit as st
import streamlit.components.v1 as components

from dashboard.components.webgpu_check import get_fallback_message


# The model ID must exactly match WebLLM's prebuilt model list
_MODEL_ID = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC"


def _build_component_html(
    context: str,
    stat_description: str,
    system_prompt: str,
) -> str:
    """Build the self-contained HTML/JS for the WebLLM chat component."""

    # Escape for safe embedding in HTML
    escaped_context = html_module.escape(context)
    escaped_stat = html_module.escape(stat_description)
    escaped_system = json.dumps(system_prompt)  # JSON-encode for JS string

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #fafafa;
        color: #333;
        padding: 12px;
    }}

    .advisor-container {{
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background: #fff;
        overflow: hidden;
    }}

    .advisor-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        font-size: 14px;
        font-weight: 600;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .status-badge {{
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 12px;
        background: rgba(255,255,255,0.2);
    }}

    .status-badge.ready {{ background: rgba(46, 204, 113, 0.3); }}
    .status-badge.loading {{ background: rgba(241, 196, 15, 0.3); }}
    .status-badge.error {{ background: rgba(231, 76, 60, 0.3); }}

    .chat-messages {{
        height: 220px;
        overflow-y: auto;
        padding: 12px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }}

    .message {{
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 13px;
        line-height: 1.5;
        max-width: 85%;
        word-wrap: break-word;
    }}

    .message.system {{
        background: #f0f0f0;
        color: #666;
        font-size: 12px;
        align-self: center;
        max-width: 95%;
        text-align: center;
    }}

    .message.user {{
        background: #667eea;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 2px;
    }}

    .message.assistant {{
        background: #f5f5f5;
        color: #333;
        align-self: flex-start;
        border-bottom-left-radius: 2px;
    }}

    .loading-bar {{
        height: 3px;
        background: #e0e0e0;
        overflow: hidden;
    }}

    .loading-bar .progress {{
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        width: 0%;
        transition: width 0.3s;
    }}

    .input-area {{
        display: flex;
        gap: 8px;
        padding: 12px;
        border-top: 1px solid #e0e0e0;
    }}

    .input-area input {{
        flex: 1;
        padding: 8px 12px;
        border: 1px solid #ddd;
        border-radius: 6px;
        font-size: 13px;
        outline: none;
    }}

    .input-area input:focus {{ border-color: #667eea; }}

    .input-area button {{
        padding: 8px 16px;
        background: #667eea;
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 13px;
        cursor: pointer;
    }}

    .input-area button:disabled {{
        background: #ccc;
        cursor: not-allowed;
    }}

    .suggested-questions {{
        padding: 8px 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }}

    .suggested-q {{
        font-size: 11px;
        padding: 4px 10px;
        border: 1px solid #ddd;
        border-radius: 14px;
        background: #fafafa;
        cursor: pointer;
        color: #555;
        transition: all 0.2s;
    }}

    .suggested-q:hover {{
        border-color: #667eea;
        color: #667eea;
        background: #f0f0ff;
    }}

    .gpu-info {{
        font-size: 10px;
        color: rgba(255,255,255,0.6);
        padding-top: 2px;
    }}
</style>
</head>
<body>
<div class="advisor-container" role="complementary" aria-label="AI Stats Advisor">
    <div class="advisor-header">
        <div>
            <div>🧠 AI Stats Advisor <span style="font-weight:400;font-size:12px;">(Qwen 2.5 1.5B — runs in your browser)</span></div>
            <div class="gpu-info" id="gpuInfo"></div>
        </div>
        <span class="status-badge loading" id="statusBadge">Initializing...</span>
    </div>

    <div class="loading-bar"><div class="progress" id="loadProgress"></div></div>

    <div class="chat-messages" id="chatMessages" role="log" aria-live="polite" aria-label="Chat messages">
        <div class="message system">
            Ask me about the statistics shown above. I'll explain what they mean
            and what you should do about them. Model loads on first question (~1.6GB download, cached after).
        </div>
    </div>

    <div class="suggested-questions" id="suggestions">
        <span class="suggested-q" role="button" tabindex="0" onclick="askQuestion(this.textContent)" onkeydown="if(event.key==='Enter')askQuestion(this.textContent)">What does this mean?</span>
        <span class="suggested-q" role="button" tabindex="0" onclick="askQuestion(this.textContent)" onkeydown="if(event.key==='Enter')askQuestion(this.textContent)">Should I worry about this?</span>
        <span class="suggested-q" role="button" tabindex="0" onclick="askQuestion(this.textContent)" onkeydown="if(event.key==='Enter')askQuestion(this.textContent)">What should I do next?</span>
        <span class="suggested-q" role="button" tabindex="0" onclick="askQuestion(this.textContent)" onkeydown="if(event.key==='Enter')askQuestion(this.textContent)">Is this normal?</span>
    </div>

    <div class="input-area">
        <input type="text" id="userInput" placeholder="Ask about the statistic above..."
               onkeydown="if(event.key==='Enter') sendMessage()">
        <button id="sendBtn" onclick="sendMessage()">Ask</button>
    </div>
</div>

<div id="fallback" style="display:none;">
    {get_fallback_message()}
</div>

<script type="module">
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

const MODEL_ID = "{_MODEL_ID}";
const SYSTEM_PROMPT = {escaped_system};
const STAT_CONTEXT = `Context about the analysis:\\n{escaped_context}\\n\\nThe specific statistic being examined:\\n{escaped_stat}`;

let engine = null;
let isLoading = false;
let isGenerating = false;

const chatEl = document.getElementById('chatMessages');
const inputEl = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const statusBadge = document.getElementById('statusBadge');
const loadProgress = document.getElementById('loadProgress');
const gpuInfo = document.getElementById('gpuInfo');
const fallback = document.getElementById('fallback');

// Check WebGPU
async function checkWebGPU() {{
    if (!navigator.gpu) {{
        showFallback('WebGPU not available');
        return false;
    }}
    try {{
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {{
            showFallback('No GPU adapter found');
            return false;
        }}
        const info = await adapter.requestAdapterInfo();
        gpuInfo.textContent = `GPU: ${{info.description || info.vendor || 'Detected'}}`;
        return true;
    }} catch(e) {{
        showFallback(e.message);
        return false;
    }}
}}

function showFallback(reason) {{
    document.querySelector('.advisor-container').style.display = 'none';
    fallback.style.display = 'block';
    console.warn('WebGPU unavailable:', reason);
}}

function setStatus(text, level) {{
    statusBadge.textContent = text;
    statusBadge.className = `status-badge ${{level}}`;
}}

function addMessage(role, text) {{
    const div = document.createElement('div');
    div.className = `message ${{role}}`;
    div.textContent = text;
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
    return div;
}}

async function initEngine() {{
    if (engine || isLoading) return;
    isLoading = true;
    setStatus('Loading model...', 'loading');

    try {{
        engine = await webllm.CreateMLCEngine(MODEL_ID, {{
            initProgressCallback: (progress) => {{
                const pct = Math.round(progress.progress * 100);
                loadProgress.style.width = pct + '%';
                setStatus(`Loading: ${{pct}}%`, 'loading');
                if (progress.text) {{
                    // Update the system message with download progress
                    const sysMsg = chatEl.querySelector('.message.system');
                    if (sysMsg && pct < 100) {{
                        sysMsg.textContent = progress.text;
                    }}
                }}
            }},
        }});
        setStatus('Ready', 'ready');
        loadProgress.style.width = '100%';
        const sysMsg = chatEl.querySelector('.message.system');
        if (sysMsg) sysMsg.textContent = 'Model loaded! Ask me anything about the statistics above.';
    }} catch(e) {{
        setStatus('Error', 'error');
        addMessage('system', `Failed to load model: ${{e.message}}`);
        console.error('WebLLM init error:', e);
    }} finally {{
        isLoading = false;
    }}
}}

async function sendMessage() {{
    const text = inputEl.value.trim();
    if (!text || isGenerating) return;

    inputEl.value = '';
    addMessage('user', text);

    // Lazy-load engine on first question
    if (!engine) {{
        await initEngine();
        if (!engine) return;
    }}

    isGenerating = true;
    sendBtn.disabled = true;
    inputEl.disabled = true;

    // Hide suggestions after first question
    document.getElementById('suggestions').style.display = 'none';

    const assistantDiv = addMessage('assistant', '');
    let fullResponse = '';

    try {{
        const messages = [
            {{ role: "system", content: SYSTEM_PROMPT + "\\n\\n" + STAT_CONTEXT }},
            {{ role: "user", content: text }},
        ];

        const completion = await engine.chat.completions.create({{
            messages: messages,
            stream: true,
            max_tokens: 400,
            temperature: 0.3,
        }});

        for await (const chunk of completion) {{
            const delta = chunk.choices[0]?.delta?.content || '';
            fullResponse += delta;
            assistantDiv.textContent = fullResponse;
            chatEl.scrollTop = chatEl.scrollHeight;
        }}
    }} catch(e) {{
        assistantDiv.textContent = `Error: ${{e.message}}`;
        console.error('Generation error:', e);
    }} finally {{
        isGenerating = false;
        sendBtn.disabled = false;
        inputEl.disabled = false;
        inputEl.focus();
    }}
}}

// Make askQuestion available globally for suggested question buttons
window.askQuestion = function(text) {{
    inputEl.value = text;
    sendMessage();
}};

// Initialize
(async () => {{
    const hasWebGPU = await checkWebGPU();
    if (hasWebGPU) {{
        setStatus('Ready to load', 'loading');
    }}
}})();
</script>
</body>
</html>
"""


def render_llm_advisor(
    context: str,
    stat_description: str,
    system_prompt: str | None = None,
    height: int = 440,
    key: str = "llm_advisor",
) -> None:
    """Render the in-browser LLM stats advisor component.

    The model downloads on first use (~1.6GB) and is cached in the
    browser's CacheStorage. Subsequent loads are instant.

    Args:
        context: Background about the analysis (model type, features, etc.)
        stat_description: The specific statistic being shown.
        system_prompt: Override the default system prompt.
        height: Component height in pixels.
        key: Unique Streamlit key.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a data science advisor helping a user understand feature "
            "engineering and data quality statistics for an ML model. "
            "Give concise, actionable advice in 2-4 sentences. Be specific to "
            "their situation — don't give generic textbook answers. "
            "If you're not sure about something, say so. "
            "Format your response as plain text, no markdown."
        )

    html_content = _build_component_html(context, stat_description, system_prompt)

    with st.expander("🧠 Ask AI about this statistic (runs locally in your browser)", expanded=False):
        components.html(html_content, height=height, scrolling=False)
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to any page with interpretations. Click "Ask AI about this statistic." Verify:
1. WebGPU detection runs (should show GPU info)
2. Clicking a suggested question triggers model download
3. Progress bar shows download progress
4. After loading, responses stream token by token
5. On second visit, model loads from cache (fast)

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add WebGPU LLM stats advisor component (Qwen 2.5 1.5B)"
```

---

## Task 3: Integrate LLM Advisor into Dashboard Pages

> **Testing note:** The inline JavaScript (350+ lines) in `render_llm_advisor()` cannot be tested with Python unit tests alone. Add a smoke test at `tests/dashboard/test_llm_advisor_js.py` that extracts the HTML string and validates: (1) all interactive elements have `id` attributes, (2) `navigator.gpu` check is present, (3) error handler functions are defined, (4) ARIA labels exist on interactive elements (chat input, send button, suggested questions). Full browser testing requires Playwright (optional but recommended for CI).

**Files:**
- Modify: `packages/dashboard/src/dashboard/pages/05_feature_importance.py`
- Modify: `packages/dashboard/src/dashboard/pages/06_feature_distributions.py`
- Modify: `packages/dashboard/src/dashboard/pages/07_feature_selection_lab.py`
- Modify: `packages/dashboard/src/dashboard/pages/08_data_quality_overview.py`
- Modify: `packages/dashboard/src/dashboard/pages/09_drift_monitor.py`
- Modify: `packages/dashboard/src/dashboard/pages/10_validation_rules.py`

The integration pattern is identical everywhere. Add this import and call at the bottom of each relevant section:

**Step 1: Add to all pages**

```python
# Add import at top of each page:
from dashboard.components.llm_advisor import render_llm_advisor
```

For **05_feature_importance.py**, add after the importance chart:

```python
# After the bar chart, at the bottom of the importance section:
render_llm_advisor(
    context=(
        f"NFL 4th-down decision model (XGBoost, 3-class classification). "
        f"Features: {', '.join(MODEL_FEATURE_COLUMNS)}. "
        f"Method: {result.method} importance."
    ),
    stat_description=(
        f"Top 5 features: "
        + ", ".join(f"{row['feature']}={row['importance']:.4f}"
                    for _, row in df_imp.head(5).iterrows())
    ),
    key="importance_advisor",
)
```

For **06_feature_distributions.py**, add inside the VIF tab and correlation tab:

```python
# In VIF tab, after VIF chart:
render_llm_advisor(
    context=f"Feature set: {', '.join(MODEL_FEATURE_COLUMNS)}",
    stat_description=f"VIF scores:\n{vif_df.to_string(index=False)}",
    key="vif_advisor",
)

# In correlation tab, after correlation pairs:
if pairs:
    render_llm_advisor(
        context=f"Feature set: {', '.join(MODEL_FEATURE_COLUMNS)}",
        stat_description=(
            "Highly correlated pairs: "
            + "; ".join(f"{a}↔{b} (r={c:.3f})" for a, b, c in pairs[:5])
        ),
        key="corr_advisor",
    )
```

For **07_feature_selection_lab.py**, add after comparison results:

```python
# After comparison metrics:
if "eval_results" in st.session_state:
    render_llm_advisor(
        context=(
            f"Comparing {all_res.n_features} features vs {sel_res.n_features} selected features. "
            f"Removed: {set(MODEL_FEATURE_COLUMNS) - set(selected)}"
        ),
        stat_description=(
            f"All features CV: {all_res.cv_mean:.4f} ± {all_res.cv_std:.4f}. "
            f"Selected CV: {sel_res.cv_mean:.4f} ± {sel_res.cv_std:.4f}. "
            f"Delta: {sel_res.cv_mean - all_res.cv_mean:+.4f}"
        ),
        key="selection_advisor",
    )
```

For **08_data_quality_overview.py**, add after the report summary:

```python
# After the three-column layout:
render_llm_advisor(
    context=f"Validating {len(data)} rows of NFL play-by-play data.",
    stat_description=f"Overall status: {report.status.value}. " + report.summary(),
    key="dq_advisor",
)
```

For **09_drift_monitor.py**, add after the overall drift summary:

```python
# After drift summary:
drifted_cols = [
    f"{name}: {d.severity.value}"
    + (f" (KS p={d.ks_pvalue:.4f})" if d.ks_pvalue else "")
    for name, d in drift_result.columns.items()
    if d.severity.value != "none"
]
if drifted_cols:
    render_llm_advisor(
        context=f"Comparing reference ({len(ref_df)} rows) vs new data ({len(new_df)} rows).",
        stat_description="Drifted columns:\n" + "\n".join(drifted_cols),
        key="drift_advisor",
    )
```

For **10_validation_rules.py**, add after rule results:

```python
# After rule results display:
if results:
    rule_summary = "; ".join(
        f"{r.rule_name}: {r.failing_rows} failing ({r.failing_pct:.1f}%)"
        for r in results[:5]
    )
    render_llm_advisor(
        context="NFL play-by-play data business rule validation.",
        stat_description=f"Rule violations: {rule_summary}",
        key="rules_advisor",
    )
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Visit each page. Verify the advisor component appears and can be expanded. Ask a test question. Verify streaming response.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): integrate WebGPU LLM advisor across all analysis pages"
```

---

## Task 4: Add Model Preloading Option + Settings

**Files:**
- Create: `packages/dashboard/src/dashboard/pages/11_settings.py`

This adds a settings page where users can pre-download the model and see GPU/memory info, rather than waiting for the first question.

**Step 1: Implement the settings page**

```python
# packages/dashboard/src/dashboard/pages/11_settings.py
"""Settings — model management, WebGPU status, and preferences."""

import streamlit as st
import streamlit.components.v1 as components

st.header("⚙️ Settings")

# --- WebGPU Status ---
st.subheader("WebGPU & AI Advisor Status")
st.markdown(
    "The AI Stats Advisor runs Qwen 2.5 1.5B directly in your browser using WebGPU. "
    "The model downloads once (~1.6GB) and is cached for future visits."
)

# Embed a small diagnostic component
diagnostic_html = """
<div id="diag" style="font-family: sans-serif; font-size: 14px; padding: 12px;
     border: 1px solid #ddd; border-radius: 8px; background: #fafafa;">
    <div id="status">Checking WebGPU...</div>
</div>
<script type="module">
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

const diag = document.getElementById('diag');
const status = document.getElementById('status');

async function diagnose() {
    let html = '';

    // WebGPU check
    if (!navigator.gpu) {
        html += '<p>❌ <strong>WebGPU:</strong> Not available in this browser</p>';
        diag.innerHTML = html;
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        html += '<p>❌ <strong>WebGPU:</strong> No GPU adapter found</p>';
        diag.innerHTML = html;
        return;
    }

    const info = await adapter.requestAdapterInfo();
    const limits = adapter.limits;

    html += `<p>✅ <strong>WebGPU:</strong> Available</p>`;
    html += `<p>🖥️ <strong>GPU:</strong> ${info.description || info.vendor || 'Unknown'}</p>`;
    html += `<p>💾 <strong>Max Buffer Size:</strong> ${(limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB</p>`;
    html += `<p>📊 <strong>Max Storage Buffer Binding:</strong> ${(limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB</p>`;

    // Check model cache
    try {
        const modelId = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";
        const caches = await caches.keys();
        const mlcCaches = caches.filter(k => k.includes('webllm') || k.includes('mlc'));
        if (mlcCaches.length > 0) {
            html += `<p>📦 <strong>Cached models:</strong> ${mlcCaches.length} cache(s) found</p>`;
        } else {
            html += `<p>📦 <strong>Cached models:</strong> None (will download on first use)</p>`;
        }
    } catch(e) {
        html += `<p>📦 <strong>Cache check:</strong> Unable to query (${e.message})</p>`;
    }

    // Available models count
    const modelCount = webllm.prebuiltAppConfig.model_list.length;
    html += `<p>🤖 <strong>WebLLM models available:</strong> ${modelCount}</p>`;

    diag.innerHTML = html;
}

diagnose().catch(e => {
    document.getElementById('status').textContent = 'Error: ' + e.message;
});
</script>
"""

components.html(diagnostic_html, height=250)

# --- Model Info ---
st.subheader("Model Information")
st.markdown("""
| Property | Value |
|----------|-------|
| **Model** | Qwen 2.5 1.5B Instruct |
| **Quantization** | q4f16_1 (4-bit weights, 16-bit activations) |
| **VRAM Required** | ~1,630 MB |
| **Context Window** | 4,096 tokens |
| **Download Size** | ~1.6 GB (one-time, cached in browser) |
| **Inference Speed** | ~10-30 tokens/sec (GPU dependent) |
| **Privacy** | All inference runs locally — no data sent to any server |
""")

# --- Tips ---
st.subheader("Tips for Best Performance")
st.markdown("""
- **Use Chrome or Edge** for the most reliable WebGPU support.
- **Close other GPU-heavy tabs** (games, video editing) before using the advisor.
- **First load takes 1-3 minutes** depending on your internet speed. After that, the model loads from cache in seconds.
- **If the model seems slow,** try closing and reopening the browser tab — this frees GPU memory from previous sessions.
- **The advisor works offline** after the first download!
""")
```

**Step 2: Manual verification**

```bash
uv run streamlit run packages/dashboard/src/dashboard/app.py
```

Navigate to Settings. Verify GPU info displays correctly.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(dashboard): add settings page with WebGPU diagnostics"
```

---

## Summary: Run Order

| Task | What it builds | Key files |
|------|---------------|-----------|
| 1 | WebGPU detection utility | `webgpu_check.py` |
| 2 | WebLLM chat component (core) | `llm_advisor.py` |
| 3 | Integration across all 6 pages | Modify pages 05-10 |
| 4 | Settings page with GPU diagnostics | `pages/11_settings.py` |

## How It Works End-to-End

1. User is on the Feature Importance page, looking at a VIF score of 8.3
2. Below the VIF chart and the static interpretation card, they see: "🧠 Ask AI about this statistic (runs locally in your browser)"
3. They expand it and see suggested questions: "What does this mean?", "Should I worry about this?"
4. They click "Should I worry about this?" (or type their own question)
5. **First time only:** WebLLM downloads Qwen 2.5 1.5B (~1.6GB). Progress bar shows download status. Model is cached in browser CacheStorage.
6. The system prompt includes the full statistical context (feature names, VIF scores, what the model is, etc.)
7. Qwen 2.5 1.5B generates a response via WebGPU, streaming token-by-token into the chat
8. User sees: "The VIF of 8.3 for score_differential is elevated, meaning about 88% of its variance is explained by other features. Since this feature is closely related to is_trailing and is_two_score_game, this is expected..."
9. They can ask follow-up questions in the same chat
10. **Next visit:** Model loads from cache in seconds, no re-download needed

## Comparison: API vs WebGPU Approach

| Aspect | Anthropic API (old Task 6) | WebGPU (this plan) |
|--------|---------------------------|-------------------|
| Setup | Need API key | Zero setup |
| Cost | Per-token billing | Free |
| Privacy | Data sent to server | 100% local |
| Quality | Sonnet-level (excellent) | 1.5B-level (good for structured Q&A) |
| Speed | 1-3 sec latency | 10-30 tok/sec after model load |
| First use | Instant | 1-3 min model download |
| Offline | No | Yes (after first download) |
| Browser support | Any | Chrome/Edge 113+ |
