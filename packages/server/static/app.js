// Configure marked with highlight.js
marked.setOptions({
  gfm: true,
  breaks: false,
});

// Custom renderer for code highlighting
const renderer = new marked.Renderer();
renderer.code = function({ text, lang }) {
  let highlighted;
  if (lang && hljs.getLanguage(lang)) {
    highlighted = hljs.highlight(text, { language: lang }).value;
  } else {
    highlighted = hljs.highlightAuto(text).value;
  }
  return `<pre><code class="hljs language-${lang || ''}">${highlighted}</code></pre>`;
};

marked.use({ renderer });

const form = document.getElementById('queryForm');
const input = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const messages = document.getElementById('messages');
const welcome = document.getElementById('welcome');
const chatArea = document.getElementById('chatArea');

// Suggestion chips
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    input.value = chip.dataset.query;
    form.dispatchEvent(new Event('submit'));
  });
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  welcome.classList.add('hidden');
  appendUserMessage(query);
  input.value = '';
  sendBtn.disabled = true;

  const loadingEl = appendLoading();

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_n: 5 }),
    });

    const data = await res.json();
    loadingEl.remove();

    if (data.chunks && data.chunks.length > 0) {
      appendResults(data.chunks, data.rewritten_query);
    } else {
      appendNoResults();
    }
  } catch (err) {
    loadingEl.remove();
    appendError(err.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
    scrollToBottom();
  }
});

function appendUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'message message-query';
  div.textContent = text;
  messages.appendChild(div);
  scrollToBottom();
}

function appendLoading() {
  const div = document.createElement('div');
  div.className = 'message loading';
  div.innerHTML = `
    <div class="loading-dots">
      <span></span><span></span><span></span>
    </div>
    <span>Searching documentation...</span>
  `;
  messages.appendChild(div);
  scrollToBottom();
  return div;
}

function renderMarkdown(text) {
  // Strip custom MDX-like tags that aren't standard markdown
  const cleaned = text
    .replace(/<\/?(?:Intro|Sandpack|YouWillLearn|DeepDive|Recap|Challenges|Solution|Hint|CodeStep|InlineToc|Illustration|Diagram|ConsoleBlock)[^>]*>/g, '')
    .replace(/\{\/\*.*?\*\/\}/g, '');  // Remove {/* comments */}
  return marked.parse(cleaned);
}

function appendResults(chunks, rewrittenQuery) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message message-results';

  // Summary bar
  const sources = [...new Set(chunks.map(c => c.source_file))];
  const topScore = Math.max(...chunks.map(c => c.score));
  const summary = document.createElement('div');
  summary.className = 'result-summary';
  const rewriteHtml = rewrittenQuery
    ? `<div class="summary-rewrite">Query Rewritten: ${escapeHtml(rewrittenQuery)}</div>`
    : '';
  summary.innerHTML = `
    <div class="summary-row">
      <span class="summary-count">${chunks.length} results</span>
      <span class="summary-sep">·</span>
      <span class="summary-sources">${sources.length} source${sources.length > 1 ? 's' : ''}</span>
      <span class="summary-sep">·</span>
      <span class="summary-score">top ${(topScore * 100).toFixed(0)}%</span>
    </div>
    ${rewriteHtml}
  `;
  wrapper.appendChild(summary);

  chunks.forEach(chunk => {
    const card = document.createElement('div');
    card.className = 'result-card';

    const heading = chunk.heading_text
      ? `<div class="result-heading">${escapeHtml(cleanHeading(chunk.heading_text))}</div>`
      : '';

    const isLong = chunk.content.length > 600;
    const displayContent = isLong ? chunk.content.slice(0, 600) : chunk.content;

    card.innerHTML = `
      <div class="result-header">
        <a class="result-source" href="#" data-filepath="${escapeAttr(chunk.source_file)}">${escapeHtml(chunk.source_file)}</a>
        <div class="result-scores">
          ${chunk.dense_score != null ? `<span class="score-tag score-dense" title="Dense search score">dense ${(chunk.dense_score * 100).toFixed(1)}%</span>` : '<span class="score-tag score-miss" title="Not found by dense search">dense —</span>'}
          ${chunk.bm25_score != null ? `<span class="score-tag score-bm25" title="BM25 score">BM25 ${chunk.bm25_score.toFixed(1)}</span>` : '<span class="score-tag score-miss" title="Not found by BM25">BM25 —</span>'}
          <span class="score-tag score-final" title="Final reranked score">${(chunk.score * 100).toFixed(1)}%</span>
        </div>
      </div>
      ${heading}
      <div class="result-content markdown-body">${renderMarkdown(displayContent)}</div>
      ${isLong ? '<button class="expand-btn">Show more</button>' : ''}
    `;

    const expandBtn = card.querySelector('.expand-btn');
    if (expandBtn) {
      let expanded = false;
      expandBtn.addEventListener('click', () => {
        const contentEl = card.querySelector('.result-content');
        expanded = !expanded;
        if (expanded) {
          contentEl.innerHTML = renderMarkdown(chunk.content);
          expandBtn.textContent = 'Show less';
        } else {
          contentEl.innerHTML = renderMarkdown(chunk.content.slice(0, 600));
          expandBtn.textContent = 'Show more';
        }
      });
    }

    wrapper.appendChild(card);
  });

  messages.appendChild(wrapper);
  scrollToBottom();
}

function cleanHeading(text) {
  // Remove {/*..*/} anchors from headings
  return text.replace(/\s*\{\/\*.*?\*\/\}\s*/g, '').replace(/^#+\s*/, '');
}

function appendNoResults() {
  const div = document.createElement('div');
  div.className = 'message no-results';
  div.textContent = 'No relevant documents found. Try rephrasing your query.';
  messages.appendChild(div);
}

function appendError(msg) {
  const div = document.createElement('div');
  div.className = 'message no-results';
  div.textContent = `Error: ${msg}`;
  messages.appendChild(div);
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
function escapeAttr(str) {
  return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Document viewer
const docOverlay = document.getElementById('docOverlay');
const docTitle = document.getElementById('docTitle');
const docContent = document.getElementById('docContent');
const docClose = document.getElementById('docClose');

// Delegate click on source file links
document.addEventListener('click', async (e) => {
  const link = e.target.closest('.result-source[data-filepath]');
  if (!link) return;
  e.preventDefault();

  const filepath = link.dataset.filepath;
  // Strip "sample-docs/" prefix if present since the API expects path relative to docs root
  const apiPath = filepath.replace(/^sample-docs\//, '');

  docTitle.textContent = filepath;
  docContent.innerHTML = '<div class="loading"><div class="loading-dots"><span></span><span></span><span></span></div><span>Loading...</span></div>';
  docOverlay.classList.add('visible');

  try {
    const res = await fetch(`/api/file?path=${encodeURIComponent(apiPath)}`);
    if (!res.ok) {
      docContent.innerHTML = `<p class="no-results">Failed to load file: ${res.statusText}</p>`;
      return;
    }
    const text = await res.text();
    docContent.innerHTML = renderMarkdown(text);
  } catch (err) {
    docContent.innerHTML = `<p class="no-results">Error: ${err.message}</p>`;
  }
});

docClose.addEventListener('click', () => {
  docOverlay.classList.remove('visible');
});

docOverlay.addEventListener('click', (e) => {
  if (e.target === docOverlay) {
    docOverlay.classList.remove('visible');
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && docOverlay.classList.contains('visible')) {
    docOverlay.classList.remove('visible');
  }
});
