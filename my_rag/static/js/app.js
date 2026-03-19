/**
 * MyRAG 前端应用 — Phase 4 (WebSocket + 对话历史)
 */

const API = "/api/v1";

const state = {
    currentKB: null,
    conversationId: null,
    isStreaming: false,
    useStream: true,
    useWebSocket: false,
    ws: null,
    knowledgeBases: [],
    conversations: [],
    currentView: "chat",
    evalDataset: [],
    evalSingleResults: {},
    evalResults: null,
    isEvaluating: false,
    savedDatasets: [],
    // HotpotQA 导入结果
    hotpotqaResult: null,
};

// ==================== 初始化 ====================

document.addEventListener("DOMContentLoaded", () => {
    marked.setOptions({
        highlight: (code, lang) => {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
    });
    loadKnowledgeBases();
});

// ==================== 知识库管理 ====================

async function loadKnowledgeBases() {
    try {
        const res = await fetch(`${API}/knowledge-bases?page=1&page_size=50`);
        const json = await res.json();
        state.knowledgeBases = json.data?.items || [];
        renderKBList();
    } catch (e) {
        console.error("加载知识库失败:", e);
    }
}

function renderKBList() {
    const container = document.getElementById("kb-list");
    if (state.knowledgeBases.length === 0) {
        container.innerHTML = '<div class="text-xs text-gray-600 text-center py-8">暂无知识库，请先创建</div>';
        return;
    }

    container.innerHTML = state.knowledgeBases.map(kb => `
        <div class="kb-card rounded-lg p-2.5 cursor-pointer border border-transparent ${state.currentKB?.id === kb.id ? 'active' : ''}"
             onclick="selectKB('${kb.id}')">
            <div class="flex items-center justify-between">
                <span class="text-sm font-medium text-gray-200 truncate">${escapeHtml(kb.name)}</span>
                <button onclick="event.stopPropagation(); deleteKB('${kb.id}')" class="text-gray-600 hover:text-red-400 transition p-0.5">
                    <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                    </svg>
                </button>
            </div>
            <div class="text-xs text-gray-500 mt-0.5">${kb.document_count} 文档 · ${kb.chunk_count} 分块</div>
        </div>
    `).join("");
}

function selectKB(kbId) {
    state.currentKB = state.knowledgeBases.find(kb => kb.id === kbId);
    state.conversationId = null;

    document.getElementById("current-kb-name").textContent = state.currentKB.name;
    document.getElementById("doc-section").classList.remove("hidden");
    document.getElementById("conv-section").classList.remove("hidden");

    const input = document.getElementById("chat-input");
    input.disabled = false;
    input.placeholder = `向「${state.currentKB.name}」提问...`;
    document.getElementById("send-btn").disabled = false;

    renderKBList();
    loadDocuments();
    loadConversations();
    clearChat();
    updateEvalButtons();
}

function showCreateKBModal() {
    document.getElementById("create-kb-modal").classList.remove("hidden");
    document.getElementById("kb-name-input").focus();
}

function hideCreateKBModal() {
    document.getElementById("create-kb-modal").classList.add("hidden");
    document.getElementById("kb-name-input").value = "";
    document.getElementById("kb-desc-input").value = "";
}

async function createKnowledgeBase() {
    const name = document.getElementById("kb-name-input").value.trim();
    const desc = document.getElementById("kb-desc-input").value.trim();
    if (!name) return showToast("请输入知识库名称", "error");

    try {
        const res = await fetch(`${API}/knowledge-bases`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, description: desc }),
        });
        const json = await res.json();
        if (json.code === 200) {
            hideCreateKBModal();
            showToast("知识库创建成功", "success");
            await loadKnowledgeBases();
            selectKB(json.data.id);
        }
    } catch (e) {
        showToast("创建失败: " + e.message, "error");
    }
}

async function deleteKB(kbId) {
    if (!confirm("确定删除此知识库？所有文档和对话记录将被清除。")) return;
    try {
        await fetch(`${API}/knowledge-bases/${kbId}`, { method: "DELETE" });
        if (state.currentKB?.id === kbId) {
            state.currentKB = null;
            state.conversationId = null;
            document.getElementById("current-kb-name").textContent = "选择一个知识库开始对话";
            document.getElementById("doc-section").classList.add("hidden");
            document.getElementById("conv-section").classList.add("hidden");
            document.getElementById("chat-input").disabled = true;
            document.getElementById("send-btn").disabled = true;
            clearChat();
        }
        showToast("已删除", "success");
        await loadKnowledgeBases();
    } catch (e) {
        showToast("删除失败", "error");
    }
}

// ==================== 文档管理 ====================

async function loadDocuments() {
    if (!state.currentKB) return;
    try {
        const res = await fetch(`${API}/knowledge-bases/${state.currentKB.id}/documents`);
        const json = await res.json();
        renderDocList(json.data?.items || []);
    } catch (e) {
        console.error("加载文档失败:", e);
    }
}

function renderDocList(docs) {
    const container = document.getElementById("doc-list");
    if (docs.length === 0) {
        container.innerHTML = '<div class="text-xs text-gray-600 text-center py-4">暂无文档</div>';
        return;
    }
    container.innerHTML = docs.map(doc => `
        <div class="doc-card flex items-center justify-between rounded-lg p-2 group">
            <div class="flex items-center gap-2 min-w-0">
                <span class="text-base">${getFileIcon(doc.filename)}</span>
                <div class="min-w-0">
                    <div class="text-xs text-gray-300 truncate" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                    <div class="text-xs text-gray-600">${formatFileSize(doc.file_size)} · <span class="status-${doc.status}">${statusText(doc.status)}</span>${doc.chunk_count > 0 ? ` · ${doc.chunk_count} 块` : ''}</div>
                </div>
            </div>
            <button onclick="deleteDoc('${doc.id}')" class="text-gray-700 hover:text-red-400 transition opacity-0 group-hover:opacity-100 p-0.5">
                <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
            </button>
        </div>
    `).join("");
}

async function uploadDocument(input) {
    if (!state.currentKB || !input.files.length) return;
    const file = input.files[0];
    const formData = new FormData();
    formData.append("file", file);

    showToast(`正在上传 ${file.name}...`, "info");

    try {
        const res = await fetch(`${API}/knowledge-bases/${state.currentKB.id}/documents`, {
            method: "POST",
            body: formData,
        });
        const json = await res.json();
        if (json.code === 200) {
            showToast("文档已上传，正在解析分块...", "info");
            await loadDocuments();
            pollDocumentStatus(json.data.id);
        } else {
            showToast(json.message || "上传失败", "error");
        }
    } catch (e) {
        showToast("上传失败: " + e.message, "error");
    }
    input.value = "";
}

function pollDocumentStatus(docId) {
    const poll = setInterval(async () => {
        try {
            const res = await fetch(`${API}/documents/${docId}/status`);
            const json = await res.json();
            const status = json.data?.status;
            if (status === "completed") {
                clearInterval(poll);
                showToast(`文档处理完成，已生成 ${json.data.chunk_count} 个分块`, "success");
                await loadDocuments();
                await loadKnowledgeBases();
            } else if (status === "failed") {
                clearInterval(poll);
                showToast("文档处理失败", "error");
                await loadDocuments();
            }
        } catch {
            clearInterval(poll);
        }
    }, 1500);
}

async function deleteDoc(docId) {
    if (!state.currentKB) return;
    try {
        await fetch(`${API}/knowledge-bases/${state.currentKB.id}/documents/${docId}`, { method: "DELETE" });
        showToast("文档已删除", "success");
        await loadDocuments();
        await loadKnowledgeBases();
    } catch (e) {
        showToast("删除失败", "error");
    }
}

async function uploadBatchDocuments(input) {
    if (!state.currentKB || !input.files.length) return;
    const files = Array.from(input.files);

    const progressEl = document.getElementById("batch-upload-progress");
    const barEl = document.getElementById("batch-upload-bar");
    const countEl = document.getElementById("batch-upload-count");
    const statusEl = document.getElementById("batch-upload-status");

    progressEl.classList.remove("hidden");
    barEl.style.width = "0%";
    statusEl.innerHTML = "";
    countEl.textContent = `0 / ${files.length}`;

    showToast(`开始批量上传 ${files.length} 个文件...`, "info");

    let done = 0;
    for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const itemEl = document.createElement("div");
        itemEl.className = "flex items-center gap-1.5 text-xs";
        itemEl.innerHTML = `<span class="eval-spinner-sm"></span><span class="text-gray-400 truncate">${escapeHtml(file.name)}</span>`;
        statusEl.appendChild(itemEl);
        statusEl.scrollTop = statusEl.scrollHeight;

        try {
            const res = await fetch(`${API}/knowledge-bases/${state.currentKB.id}/documents`, {
                method: "POST",
                body: formData,
            });
            const json = await res.json();
            if (json.code === 200) {
                itemEl.innerHTML = `<span class="text-emerald-400">✓</span><span class="text-gray-400 truncate">${escapeHtml(file.name)}</span>`;
            } else {
                itemEl.innerHTML = `<span class="text-red-400">✗</span><span class="text-gray-500 truncate">${escapeHtml(file.name)}: ${escapeHtml(json.message || "失败")}</span>`;
            }
        } catch (e) {
            itemEl.innerHTML = `<span class="text-red-400">✗</span><span class="text-gray-500 truncate">${escapeHtml(file.name)}: ${escapeHtml(e.message)}</span>`;
        }

        done++;
        barEl.style.width = `${Math.round(done / files.length * 100)}%`;
        countEl.textContent = `${done} / ${files.length}`;
    }

    showToast(`批量上传完成：${files.length} 个文件`, "success");
    await loadDocuments();
    await loadKnowledgeBases();
    input.value = "";

    setTimeout(() => progressEl.classList.add("hidden"), 5000);
}

// ==================== 对话历史 ====================

async function loadConversations() {
    if (!state.currentKB) return;
    try {
        const res = await fetch(`${API}/chat/conversations?knowledge_base_id=${state.currentKB.id}`);
        const json = await res.json();
        state.conversations = json.data || [];
        renderConvList();
    } catch (e) {
        console.error("加载对话历史失败:", e);
    }
}

function renderConvList() {
    const container = document.getElementById("conv-list");
    if (state.conversations.length === 0) {
        container.innerHTML = '<div class="text-xs text-gray-600 text-center py-4">暂无对话</div>';
        return;
    }
    container.innerHTML = state.conversations.map(conv => `
        <div class="flex items-center justify-between rounded-lg p-2 cursor-pointer hover:bg-gray-800/50 group ${state.conversationId === conv.id ? 'bg-gray-800/70' : ''}"
             onclick="selectConversation('${conv.id}')">
            <div class="min-w-0 flex-1">
                <div class="text-xs text-gray-300 truncate">${escapeHtml(conv.title || '未命名对话')}</div>
                <div class="text-xs text-gray-600">${conv.message_count} 条消息</div>
            </div>
            <button onclick="event.stopPropagation(); deleteConversation('${conv.id}')" class="text-gray-700 hover:text-red-400 transition opacity-0 group-hover:opacity-100 p-0.5 shrink-0">
                <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
            </button>
        </div>
    `).join("");
}

async function selectConversation(convId) {
    state.conversationId = convId;
    renderConvList();
    // TODO: load conversation messages and display
}

function startNewConversation() {
    clearChat();
    showToast("已开始新对话", "info");
}

async function deleteConversation(convId) {
    try {
        await fetch(`${API}/chat/conversations/${convId}`, { method: "DELETE" });
        if (state.conversationId === convId) {
            state.conversationId = null;
            clearChat();
        }
        await loadConversations();
        showToast("对话已删除", "success");
    } catch (e) {
        showToast("删除失败", "error");
    }
}

// ==================== WebSocket ====================

function toggleWebSocket(checkbox) {
    state.useWebSocket = checkbox.checked;
    if (state.useWebSocket) {
        connectWebSocket();
    } else {
        disconnectWebSocket();
    }
}

function connectWebSocket() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) return;

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${location.host}${API}/ws/chat`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        showToast("WebSocket 已连接", "success");
        document.getElementById("status-text").textContent = "WebSocket 已连接";
        document.getElementById("status-dot").className = "w-2 h-2 rounded-full bg-blue-500";
    };

    state.ws.onclose = () => {
        document.getElementById("status-text").textContent = "系统就绪";
        document.getElementById("status-dot").className = "w-2 h-2 rounded-full bg-emerald-500";
    };

    state.ws.onerror = () => {
        showToast("WebSocket 连接失败", "error");
        state.useWebSocket = false;
        document.getElementById("ws-toggle").checked = false;
    };
}

function disconnectWebSocket() {
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }
}

async function sendViaWebSocket(query) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        showToast("WebSocket 未连接", "error");
        return;
    }

    document.getElementById("welcome").classList.add("hidden");
    document.getElementById("messages").classList.remove("hidden");
    appendMessage("user", query);

    const assistantEl = appendMessage("assistant", "");
    const contentEl = assistantEl.querySelector(".msg-content");
    contentEl.innerHTML = '<div class="dot-pulse"><span></span><span></span><span></span></div>';

    state.isStreaming = true;
    updateSendButton();

    let fullAnswer = "";
    let sources = [];

    state.ws.send(JSON.stringify({
        type: "chat",
        query: query,
        knowledge_base_id: state.currentKB.id,
        conversation_id: state.conversationId,
    }));

    return new Promise((resolve) => {
        const handler = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === "retrieval") {
                    sources = data.documents || [];
                } else if (data.type === "token") {
                    fullAnswer += data.content;
                    contentEl.innerHTML = renderMarkdown(fullAnswer);
                    contentEl.classList.add("typing-cursor");
                    scrollToBottom();
                } else if (data.type === "done") {
                    state.conversationId = data.conversation_id || state.conversationId;
                    contentEl.classList.remove("typing-cursor");
                    contentEl.innerHTML = renderMarkdown(fullAnswer);
                    if (sources.length > 0) appendSources(assistantEl, sources);

                    state.isStreaming = false;
                    updateSendButton();
                    scrollToBottom();
                    loadConversations();

                    state.ws.removeEventListener("message", handler);
                    resolve();
                } else if (data.type === "error") {
                    contentEl.innerHTML = `<span class="text-red-400">${escapeHtml(data.message)}</span>`;
                    state.isStreaming = false;
                    updateSendButton();
                    state.ws.removeEventListener("message", handler);
                    resolve();
                }
            } catch {}
        };

        state.ws.addEventListener("message", handler);
    });
}

// ==================== 对话（SSE / WebSocket 统一入口） ====================

function toggleStream(checkbox) {
    state.useStream = checkbox.checked;
}

async function sendMessage() {
    if (state.isStreaming || !state.currentKB) return;

    const input = document.getElementById("chat-input");
    const query = input.value.trim();
    if (!query) return;

    input.value = "";
    autoResize(input);

    if (state.useWebSocket) {
        await sendViaWebSocket(query);
        return;
    }

    document.getElementById("welcome").classList.add("hidden");
    document.getElementById("messages").classList.remove("hidden");

    appendMessage("user", query);

    const assistantEl = appendMessage("assistant", "");
    const contentEl = assistantEl.querySelector(".msg-content");
    contentEl.innerHTML = '<div class="dot-pulse"><span></span><span></span><span></span></div>';

    state.isStreaming = true;
    updateSendButton();

    try {
        const res = await fetch(`${API}/chat/completions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query,
                knowledge_base_id: state.currentKB.id,
                conversation_id: state.conversationId,
                stream: state.useStream,
            }),
        });

        if (state.useStream) {
            await handleStreamResponse(res, contentEl, assistantEl);
        } else {
            await handleNormalResponse(res, contentEl, assistantEl);
        }
    } catch (e) {
        contentEl.innerHTML = `<span class="text-red-400">请求失败: ${escapeHtml(e.message)}</span>`;
    }

    state.isStreaming = false;
    updateSendButton();
    scrollToBottom();
    loadConversations();
}

async function handleStreamResponse(res, contentEl, assistantEl) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullAnswer = "";
    let sources = [];
    let buffer = "";

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
                const event = JSON.parse(line.slice(6));

                if (event.type === "retrieval") {
                    sources = event.documents || [];
                } else if (event.type === "token") {
                    fullAnswer += event.content;
                    contentEl.innerHTML = renderMarkdown(fullAnswer);
                    contentEl.classList.add("typing-cursor");
                    scrollToBottom();
                } else if (event.type === "done") {
                    state.conversationId = event.conversation_id || state.conversationId;
                    if (event.usage) {
                        document.getElementById("token-info").textContent =
                            `Tokens: ${event.usage.prompt_tokens} + ${event.usage.completion_tokens} = ${event.usage.total_tokens}`;
                    }
                }
            } catch {}
        }
    }

    contentEl.classList.remove("typing-cursor");
    contentEl.innerHTML = renderMarkdown(fullAnswer);

    if (sources.length > 0) {
        appendSources(assistantEl, sources);
    }
}

async function handleNormalResponse(res, contentEl, assistantEl) {
    const json = await res.json();
    const data = json.data;

    if (!data) {
        contentEl.innerHTML = `<span class="text-red-400">请求失败: ${escapeHtml(json.message || "未知错误")}</span>`;
        return;
    }

    state.conversationId = data.conversation_id || state.conversationId;
    contentEl.innerHTML = renderMarkdown(data.answer);

    if (data.usage) {
        document.getElementById("token-info").textContent =
            `Tokens: ${data.usage.prompt_tokens} + ${data.usage.completion_tokens} = ${data.usage.total_tokens}`;
    }

    if (data.sources && data.sources.length > 0) {
        appendSources(assistantEl, data.sources);
    }
}

function appendMessage(role, content) {
    const container = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = `fade-in flex ${role === "user" ? "justify-end" : "justify-start"}`;

    const isUser = role === "user";
    div.innerHTML = `
        <div class="max-w-[85%] ${isUser ? '' : 'w-full'}">
            <div class="flex items-center gap-2 mb-1.5 ${isUser ? 'justify-end' : ''}">
                <span class="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${isUser ? 'bg-brand-600' : 'bg-emerald-600'}">
                    ${isUser ? "U" : "A"}
                </span>
                <span class="text-xs text-gray-500">${isUser ? "你" : "MyRAG 助手"}</span>
            </div>
            <div class="${isUser ? 'msg-user' : 'msg-assistant'} rounded-xl px-4 py-3">
                <div class="msg-content markdown-body">${isUser ? escapeHtml(content) : content}</div>
            </div>
            <div class="sources-area"></div>
        </div>
    `;

    container.appendChild(div);
    scrollToBottom();
    return div;
}

function appendSources(msgEl, sources) {
    const area = msgEl.querySelector(".sources-area");
    area.innerHTML = `
        <div class="mt-2">
            <div class="sources-toggle flex items-center gap-1 text-xs text-gray-500 hover:text-gray-400" onclick="this.nextElementSibling.classList.toggle('open')">
                <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                </svg>
                引用来源 (${sources.length})
            </div>
            <div class="sources-content mt-1 space-y-1.5">
                ${sources.map((s, i) => `
                    <div class="bg-gray-800/50 border border-gray-700/50 rounded-lg p-2.5 text-xs">
                        <div class="flex items-center justify-between mb-1">
                            <span class="text-gray-400 font-medium">${escapeHtml(s.source)}</span>
                            <span class="text-brand-400">${(s.score * 100).toFixed(1)}%</span>
                        </div>
                        <p class="text-gray-500 line-clamp-2">${escapeHtml(s.content)}</p>
                    </div>
                `).join("")}
            </div>
        </div>
    `;
}

function clearChat() {
    state.conversationId = null;
    document.getElementById("messages").innerHTML = "";
    document.getElementById("messages").classList.add("hidden");
    document.getElementById("welcome").classList.remove("hidden");
    document.getElementById("token-info").textContent = "";
    renderConvList();
}

// ==================== 工具函数 ====================

function handleInputKeydown(event) {
    if (event.ctrlKey && event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
}

function updateSendButton() {
    const btn = document.getElementById("send-btn");
    btn.disabled = state.isStreaming || !state.currentKB;
}

function scrollToBottom() {
    const area = document.getElementById("chat-area");
    area.scrollTop = area.scrollHeight;
}

function renderMarkdown(text) {
    return marked.parse(text);
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function getFileIcon(filename) {
    const ext = filename.split(".").pop()?.toLowerCase();
    const icons = { pdf: "📕", docx: "📘", doc: "📘", md: "📝", txt: "📄", html: "🌐" };
    return icons[ext] || "📄";
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function statusText(status) {
    return { pending: "待处理", processing: "处理中", completed: "已完成", failed: "失败" }[status] || status;
}

// ==================== 视图切换 ====================

function switchView(view) {
    state.currentView = view;
    document.getElementById("tab-chat").classList.toggle("active", view === "chat");
    document.getElementById("tab-eval").classList.toggle("active", view === "eval");
    document.getElementById("tab-metrics").classList.toggle("active", view === "metrics");

    const chatArea = document.getElementById("chat-area");
    const evalPanel = document.getElementById("eval-panel");
    const metricsPanel = document.getElementById("metrics-panel");
    const inputArea = document.getElementById("chat-input-area");

    chatArea.classList.add("hidden");
    evalPanel.classList.add("hidden");
    metricsPanel.classList.add("hidden");
    inputArea.classList.add("hidden");

    if (view === "chat") {
        chatArea.classList.remove("hidden");
        inputArea.classList.remove("hidden");
    } else if (view === "eval") {
        evalPanel.classList.remove("hidden");
        loadSavedDatasets();
    } else if (view === "metrics") {
        metricsPanel.classList.remove("hidden");
        fetchAndRenderMetrics();
        startMetricsAutoRefresh();
    }

    if (view !== "metrics") {
        stopMetricsAutoRefresh();
    }

    updateEvalButtons();
}

function updateEvalButtons() {
    const hasKB = !!state.currentKB;
    const hasData = state.evalDataset.length > 0;
    const genBtn = document.getElementById("gen-dataset-btn");
    const batchBtn = document.getElementById("run-batch-btn");
    if (genBtn) genBtn.disabled = !hasKB || state.isEvaluating;
    if (batchBtn) batchBtn.disabled = !hasKB || !hasData || state.isEvaluating;
}

// ==================== 已有数据集管理 ====================

async function loadSavedDatasets() {
    const container = document.getElementById("saved-datasets-list");
    const info = document.getElementById("saved-datasets-info");

    try {
        const res = await fetch(`${API}/evaluations/datasets`);
        const json = await res.json();
        state.savedDatasets = json.data || [];

        if (state.savedDatasets.length === 0) {
            info.textContent = "暂无已保存的数据集";
            container.innerHTML = '<div class="text-xs text-gray-600 text-center py-4">暂无已保存的数据集，请先生成</div>';
            return;
        }

        info.textContent = `共 ${state.savedDatasets.length} 个数据集`;
        container.innerHTML = state.savedDatasets.map(ds => `
            <div class="flex items-center justify-between bg-gray-800/50 border border-gray-700/50 rounded-lg px-3 py-2.5 group hover:border-brand-600/50 transition cursor-pointer"
                 onclick="selectSavedDataset('${escapeHtml(ds.name)}')">
                <div class="flex items-center gap-3 min-w-0">
                    <div class="w-8 h-8 rounded-lg bg-brand-600/15 flex items-center justify-center shrink-0">
                        <svg class="w-4 h-4 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                    </div>
                    <div class="min-w-0">
                        <div class="text-xs font-medium text-gray-200 truncate">${escapeHtml(ds.name)}</div>
                        <div class="text-xs text-gray-500">${ds.total_samples} 条样本 · ${ds.file_size_kb} KB</div>
                    </div>
                </div>
                <div class="flex items-center gap-1.5 shrink-0">
                    <span class="text-xs text-brand-500 opacity-0 group-hover:opacity-100 transition">加载</span>
                    <button onclick="event.stopPropagation(); deleteSavedDataset('${escapeHtml(ds.name)}')"
                        class="text-gray-700 hover:text-red-400 transition opacity-0 group-hover:opacity-100 p-0.5">
                        <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
                    </button>
                </div>
            </div>
        `).join("");
    } catch (e) {
        info.textContent = "加载失败";
        container.innerHTML = `<div class="text-xs text-red-400 text-center py-4">加载失败: ${escapeHtml(e.message)}</div>`;
    }
}

async function selectSavedDataset(name) {
    showToast(`正在加载数据集「${name}」...`, "info");
    try {
        const res = await fetch(`${API}/evaluations/datasets/${encodeURIComponent(name)}`);
        const json = await res.json();

        if (json.code === 200 && json.data) {
            // 自动选中对应的知识库
            const kbId = json.data.knowledge_base_id;
            if (kbId && (!state.currentKB || state.currentKB.id !== kbId)) {
                const targetKB = state.knowledgeBases.find(kb => kb.id === kbId);
                if (targetKB) {
                    state.currentKB = targetKB;
                    document.getElementById("current-kb-name").textContent = targetKB.name;
                    document.getElementById("doc-section").classList.remove("hidden");
                    document.getElementById("conv-section").classList.remove("hidden");
                    document.getElementById("chat-input").disabled = false;
                    document.getElementById("chat-input").placeholder = `向「${targetKB.name}」提问...`;
                    document.getElementById("send-btn").disabled = false;
                    renderKBList();
                    loadDocuments();
                    loadConversations();
                }
            }

            state.evalSingleResults = {};
            state.evalResults = null;
            state.evalDataset = json.data.samples.map(s => ({
                question: s.question,
                ground_truth: s.ground_truth,
                metadata: s.metadata || {},
            }));
            renderDatasetTable();
            updateEvalButtons();

            document.getElementById("eval-results-section").classList.add("hidden");
            showToast(`已加载「${name}」，共 ${json.data.total_samples} 条样本`, "success");
        } else {
            showToast(json.message || "加载失败", "error");
        }
    } catch (e) {
        showToast("加载失败: " + e.message, "error");
    }
}

async function deleteSavedDataset(name) {
    if (!confirm(`确定删除数据集「${name}」？此操作不可恢复。`)) return;
    try {
        const res = await fetch(`${API}/evaluations/datasets/${encodeURIComponent(name)}`, { method: "DELETE" });
        const json = await res.json();
        if (json.code === 200) {
            showToast(`数据集「${name}」已删除`, "success");
            await loadSavedDatasets();
        } else {
            showToast(json.message || "删除失败", "error");
        }
    } catch (e) {
        showToast("删除失败: " + e.message, "error");
    }
}

// ==================== HotpotQA 数据集导入 ====================

async function importHotpotQA() {
    const btn = document.getElementById("import-hotpotqa-btn");
    const loadingEl = document.getElementById("hotpotqa-loading");
    const resultEl = document.getElementById("hotpotqa-result");

    const numSamples = parseInt(document.getElementById("hq-num-samples").value) || 20;
    const split = document.getElementById("hq-split").value;
    const onlySupporting = document.getElementById("hq-only-supporting").value === "true";
    const exportDocx = document.getElementById("hq-export-docx").checked;
    const mergeDocx = document.getElementById("hq-merge-docx").checked;
    const autoIngest = document.getElementById("hq-auto-ingest").checked;
    const datasetName = document.getElementById("hq-dataset-name").value.trim();
    const kbId = state.currentKB?.id || "";

    if (autoIngest && !kbId) {
        showToast("已勾选「自动写入知识库」，请先在左侧选择一个知识库", "error");
        return;
    }

    btn.disabled = true;
    const loadingTextEl = loadingEl.querySelector(".text-xs.text-gray-300");
    if (loadingTextEl) loadingTextEl.textContent = "正在从 HuggingFace 拉取数据集...";
    loadingEl.classList.remove("hidden");
    resultEl.classList.add("hidden");

    try {
        const res = await fetch(`${API}/evaluations/import-hotpotqa`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                num_samples: numSamples,
                split,
                config_name: "distractor",
                only_supporting: onlySupporting,
                export_docx: exportDocx,
                merge_docx: mergeDocx,
                auto_ingest: autoIngest,
                dataset_name: datasetName,
                knowledge_base_id: kbId,
            }),
        });
        const json = await res.json();

        if (json.code === 200 && json.data) {
            state.hotpotqaResult = json.data;
            // 先渲染基础结果（ingest 还在后台跑）
            renderHotpotQAResult(json.data);
            showToast(`HotpotQA 导入成功：${json.data.total_samples} 条样本，${json.data.total_articles} 篇文章`, "success");
            await loadSavedDatasets();

            // 如果有后台 ingest 任务，启动轮询
            if (json.data.ingest_task_id) {
                pollIngestProgress(json.data.ingest_task_id, json.data.ingest_total);
            }
        } else {
            showToast(json.message || "导入失败", "error");
        }
    } catch (e) {
        showToast("导入失败: " + e.message, "error");
    }

    btn.disabled = false;
    loadingEl.classList.add("hidden");
}

async function pollIngestProgress(taskId, total) {
    const ingestStatsEl = document.getElementById("hq-ingest-stats");
    ingestStatsEl.classList.remove("hidden");

    let interval = 2000;  // 初始 2 秒轮询一次
    let attempts = 0;
    const maxAttempts = 120;  // 最多轮询 120 次（约 4 分钟）

    while (attempts < maxAttempts) {
        await new Promise(r => setTimeout(r, interval));
        attempts++;

        try {
            const res = await fetch(`${API}/evaluations/ingest-status/${encodeURIComponent(taskId)}`);
            const json = await res.json();
            if (json.code !== 200 || !json.data) break;

            const p = json.data;
            const pct = total > 0 ? Math.round((p.done / total) * 100) : 0;

            if (p.status === "running") {
                ingestStatsEl.innerHTML =
                    `<span class="text-blue-400">⏳ 正在写入知识库：${p.done}/${p.total} 篇完成（${pct}%）· 已耗时 ${p.elapsed_s}s</span>`;
                // 随着进度推进，轮询间隔逐渐拉长（避免频繁请求）
                interval = Math.min(interval + 1000, 5000);
            } else if (p.status === "completed") {
                const failPart = p.failed > 0
                    ? `<span class="text-red-400 ml-2">· ${p.failed} 篇失败</span>`
                    : "";
                ingestStatsEl.innerHTML =
                    `<span class="text-emerald-400">✓ 已写入知识库：${p.done} 篇文章（chunk + 向量化完成，耗时 ${p.elapsed_s}s）</span>${failPart}`;
                showToast(`知识库写入完成：${p.done} 篇文章`, "success");
                break;
            } else if (p.status === "failed") {
                ingestStatsEl.innerHTML =
                    `<span class="text-red-400">✗ 写入失败：${p.error || "未知错误"}</span>`;
                showToast("知识库写入失败", "error");
                break;
            }
        } catch (e) {
            // 网络异常，继续重试
        }
    }
}

function renderHotpotQAResult(data) {
    const resultEl = document.getElementById("hotpotqa-result");
    resultEl.classList.remove("hidden");

    document.getElementById("hq-result-title").textContent = `已导入：${data.dataset_name}`;
    document.getElementById("hq-result-stats").textContent =
        `${data.total_samples} 条评估样本 · ${data.total_articles} 篇 context 文章` +
        (data.docx_dir ? ` · docx 目录: ${data.docx_dir}` : "");

    // ingest 进度由 pollIngestProgress 动态更新，这里只做初始化
    const ingestStatsEl = document.getElementById("hq-ingest-stats");
    if (data.ingest_task_id && data.ingest_total > 0) {
        ingestStatsEl.classList.remove("hidden");
        ingestStatsEl.innerHTML =
            `<span class="text-blue-400">⏳ 正在后台写入知识库（${data.ingest_total} 篇文章），chunk + 向量化中...</span>`;
    } else {
        ingestStatsEl.classList.add("hidden");
    }

    // 显示原始 JSON 路径
    const rawPathEl = document.getElementById("hq-raw-json-path");
    if (data.raw_json_path) {
        rawPathEl.textContent = `完整原始 JSON：${data.raw_json_path}`;
        rawPathEl.classList.remove("hidden");
    } else {
        rawPathEl.classList.add("hidden");
    }

    // 显示 docx 文件列表
    const docxListEl = document.getElementById("hq-docx-list");
    const docxFilesEl = document.getElementById("hq-docx-files");
    if (data.docx_files && data.docx_files.length > 0) {
        docxListEl.classList.remove("hidden");
        docxFilesEl.innerHTML = data.docx_files.map(fp => {
            const name = fp.split(/[\\/]/).pop();
            return `<span class="inline-flex items-center gap-1 bg-gray-700/50 border border-gray-600/50 rounded-md px-2 py-0.5 text-xs text-gray-300" title="${escapeHtml(fp)}">
                <span>📘</span><span class="truncate max-w-[120px]">${escapeHtml(name)}</span>
            </span>`;
        }).join("");
    } else {
        docxListEl.classList.add("hidden");
    }

    // 显示样本预览（含完整 HotpotQA 元信息）
    const samplesPreviewEl = document.getElementById("hq-samples-preview");
    const samplesListEl = document.getElementById("hq-samples-list");
    if (data.samples && data.samples.length > 0) {
        samplesPreviewEl.classList.remove("hidden");
        samplesListEl.innerHTML = data.samples.map((s, i) => {
            const meta = s.metadata || {};
            const typeBadge = meta.type
                ? `<span class="px-1.5 py-0.5 rounded text-[10px] font-medium ${meta.type === 'bridge' ? 'bg-violet-900/50 text-violet-300' : 'bg-amber-900/50 text-amber-300'}">${meta.type}</span>`
                : "";
            const levelBadge = meta.level
                ? `<span class="px-1.5 py-0.5 rounded text-[10px] font-medium ${meta.level === 'hard' ? 'bg-red-900/50 text-red-300' : meta.level === 'medium' ? 'bg-yellow-900/50 text-yellow-300' : 'bg-green-900/50 text-green-300'}">${meta.level}</span>`
                : "";
            const supportingHtml = meta.supporting_titles?.length
                ? `<div class="text-[10px] text-gray-600 mt-1">支撑文章：${meta.supporting_titles.map(t => `<span class="text-gray-500">${escapeHtml(t)}</span>`).join(" · ")}</div>`
                : "";
            return `
                <div class="bg-gray-900/60 border border-gray-700/50 rounded-lg px-3 py-2">
                    <div class="flex items-start gap-2 mb-1">
                        <span class="text-xs text-gray-600 shrink-0">Q${i + 1}</span>
                        <div class="flex-1">
                            <div class="flex items-center gap-1.5 mb-0.5 flex-wrap">
                                ${typeBadge}${levelBadge}
                            </div>
                            <div class="text-xs text-gray-300">${escapeHtml(s.question)}</div>
                        </div>
                    </div>
                    <div class="text-xs text-emerald-400 font-medium pl-5">答案：${escapeHtml(s.ground_truth)}</div>
                    ${supportingHtml}
                </div>
            `;
        }).join("");
    } else {
        samplesPreviewEl.classList.add("hidden");
    }

    // 根据是否有 docx 控制批量上传按钮
    const uploadBtn = document.getElementById("hq-batch-upload-btn");
    if (!data.docx_files || data.docx_files.length === 0) {
        uploadBtn.classList.add("hidden");
    } else {
        uploadBtn.classList.remove("hidden");
    }
}

function loadHotpotQADataset() {
    if (!state.hotpotqaResult) return;
    const dsName = state.hotpotqaResult.dataset_name;
    selectSavedDataset(dsName);
}

async function uploadHotpotQADocs() {
    if (!state.hotpotqaResult || !state.hotpotqaResult.docx_files?.length) {
        showToast("没有可上传的 docx 文件", "error");
        return;
    }
    if (!state.currentKB) {
        showToast("请先选择知识库", "error");
        return;
    }

    const files = state.hotpotqaResult.docx_files;
    const btn = document.getElementById("hq-batch-upload-btn");
    btn.disabled = true;
    btn.innerHTML = `<span class="eval-spinner-sm"></span> 上传中...`;

    showToast(`正在批量上传 ${files.length} 个 docx 文件到知识库...`, "info");

    try {
        // 通过服务端路径批量上传（调用后端批量接口）
        // 由于文件在服务端，我们需要通过特殊接口处理
        // 这里改为提示用户手动选择文件，或调用服务端路径上传接口
        const res = await fetch(`${API}/evaluations/upload-hotpotqa-docs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                knowledge_base_id: state.currentKB.id,
                docx_dir: state.hotpotqaResult.docx_dir,
            }),
        });
        const json = await res.json();

        if (json.code === 200 && json.data) {
            const d = json.data;
            showToast(`上传完成：成功 ${d.success}，失败 ${d.failed}，跳过 ${d.skipped}`, d.failed > 0 ? "info" : "success");
            await loadDocuments();
            await loadKnowledgeBases();
        } else {
            showToast(json.message || "上传失败", "error");
        }
    } catch (e) {
        showToast("上传失败: " + e.message, "error");
    }

    btn.disabled = false;
    btn.innerHTML = `<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg> 批量上传 docx 到知识库`;
}

// ==================== 评估数据集生成 ====================

async function generateDataset() {
    if (!state.currentKB || state.isEvaluating) return;

    const qpc = parseInt(document.getElementById("gen-qpc").value) || 2;
    const max = parseInt(document.getElementById("gen-max").value) || 20;
    const strategy = document.getElementById("gen-strategy").value;
    const name = document.getElementById("gen-name").value.trim();

    state.isEvaluating = true;
    updateEvalButtons();
    showToast("正在生成评估数据集，请稍候...", "info");

    try {
        const res = await fetch(`${API}/evaluations/generate-dataset`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                knowledge_base_id: state.currentKB.id,
                dataset_name: name,
                num_questions_per_chunk: qpc,
                max_chunks: max,
                sampling_strategy: strategy,
                include_multi_chunk: true,
                save_to_file: true,
            }),
        });
        const json = await res.json();
        if (json.code === 200 && json.data) {
            state.evalSingleResults = {};
            state.evalDataset = json.data.samples.map(s => ({
                question: s.question,
                ground_truth: s.ground_truth,
                metadata: s.metadata || {},
            }));
            renderDatasetTable();
            loadSavedDatasets();
            showToast(`已生成 ${json.data.total_samples} 条评估样本`, "success");
        } else {
            showToast(json.message || "生成失败", "error");
        }
    } catch (e) {
        showToast("生成失败: " + e.message, "error");
    }

    state.isEvaluating = false;
    updateEvalButtons();
}

// ==================== 数据集表格 ====================

function renderDatasetTable() {
    const tbody = document.getElementById("dataset-tbody");
    const tableWrap = document.getElementById("dataset-table-wrap");
    const empty = document.getElementById("dataset-empty");
    const info = document.getElementById("dataset-info");

    if (state.evalDataset.length === 0) {
        tableWrap.classList.add("hidden");
        empty.classList.remove("hidden");
        info.textContent = "尚未生成数据集";
        return;
    }

    tableWrap.classList.remove("hidden");
    empty.classList.add("hidden");

    const doneCount = Object.keys(state.evalSingleResults).filter(k => state.evalSingleResults[k]?.status === "done").length;
    const runningCount = Object.keys(state.evalSingleResults).filter(k => state.evalSingleResults[k]?.status === "running").length;
    let infoText = `共 ${state.evalDataset.length} 条样本`;
    if (doneCount > 0) infoText += ` · ${doneCount} 条已评估`;
    if (runningCount > 0) infoText += ` · ${runningCount} 条评估中`;
    info.textContent = infoText;

    // 检测是否有 HotpotQA 元数据
    const hasHotpotMeta = state.evalDataset.some(s => s.metadata?.hotpotqa_id);

    tbody.innerHTML = state.evalDataset.map((s, i) => {
        const result = state.evalSingleResults[i];
        const statusHtml = buildSampleStatus(i, result);
        const actionsHtml = buildSampleActions(i, result);
        const metaHtml = hasHotpotMeta ? buildHotpotMetaBadges(s.metadata) : "";
        return `
            <tr>
                <td class="text-gray-500">${i + 1}</td>
                <td class="max-w-xs">
                    <div class="truncate" title="${escapeHtml(s.question)}">${escapeHtml(s.question)}</div>
                    ${metaHtml}
                </td>
                <td class="max-w-sm">
                    <div class="truncate text-gray-400" title="${escapeHtml(s.ground_truth)}">${escapeHtml(s.ground_truth) || '<span class="text-gray-600 italic">未设置</span>'}</div>
                </td>
                <td class="text-center">${statusHtml}</td>
                <td class="text-center">${actionsHtml}</td>
            </tr>
        `;
    }).join("");
}

function buildSampleStatus(index, result) {
    if (!result) return '<span class="text-xs text-gray-600">待评估</span>';
    if (result.status === "running") {
        return '<span class="inline-flex items-center gap-1 text-xs text-blue-400"><span class="eval-spinner-sm"></span>评估中</span>';
    }
    if (result.status === "error") {
        return '<span class="text-xs text-red-400" title="' + escapeHtml(result.error || '') + '">失败</span>';
    }
    const overall = result.data?.scores?.overall || 0;
    return scorePill(overall);
}

function buildSampleActions(index, result) {
    const evalBtn = result?.status === "running"
        ? `<button disabled class="eval-row-btn disabled text-xs px-2 py-1 rounded-md">评估中...</button>`
        : `<button onclick="runSingleEval(${index})" class="eval-row-btn text-xs px-2 py-1 rounded-md transition">${result?.status === "done" ? '重新评估' : '评估'}</button>`;

    const detailBtn = result?.status === "done"
        ? `<button onclick="showEvalDetail(${index})" class="eval-detail-btn text-xs px-2 py-1 rounded-md transition">详情</button>`
        : '';

    const deleteBtn = `<button onclick="removeDatasetSample(${index})" class="text-gray-600 hover:text-red-400 transition text-xs px-1">删除</button>`;

    return `<div class="flex items-center justify-center gap-1">${evalBtn}${detailBtn}${deleteBtn}</div>`;
}

function buildHotpotMetaBadges(metadata) {
    if (!metadata) return "";
    const parts = [];
    if (metadata.type) {
        const typeColor = metadata.type === "bridge" ? "bg-blue-900/40 text-blue-400 border-blue-800/50" : "bg-purple-900/40 text-purple-400 border-purple-800/50";
        parts.push(`<span class="inline-block border rounded px-1 py-0 text-[10px] ${typeColor}">${escapeHtml(metadata.type)}</span>`);
    }
    if (metadata.level) {
        const lvlColor = metadata.level === "easy" ? "bg-emerald-900/40 text-emerald-400 border-emerald-800/50"
            : metadata.level === "medium" ? "bg-amber-900/40 text-amber-400 border-amber-800/50"
            : "bg-red-900/40 text-red-400 border-red-800/50";
        parts.push(`<span class="inline-block border rounded px-1 py-0 text-[10px] ${lvlColor}">${escapeHtml(metadata.level)}</span>`);
    }
    if (metadata.supporting_titles?.length) {
        const titles = metadata.supporting_titles.slice(0, 2).map(t => escapeHtml(t)).join(", ");
        parts.push(`<span class="text-[10px] text-gray-600" title="支撑文章: ${escapeHtml(metadata.supporting_titles.join(', '))}">📎 ${titles}${metadata.supporting_titles.length > 2 ? '...' : ''}</span>`);
    }
    return parts.length ? `<div class="flex flex-wrap items-center gap-1 mt-0.5">${parts.join("")}</div>` : "";
}

function removeDatasetSample(index) {
    state.evalDataset.splice(index, 1);
    const newResults = {};
    for (const [k, v] of Object.entries(state.evalSingleResults)) {
        const ki = parseInt(k);
        if (ki < index) newResults[ki] = v;
        else if (ki > index) newResults[ki - 1] = v;
    }
    state.evalSingleResults = newResults;
    renderDatasetTable();
    updateEvalButtons();
}

function addManualSample() {
    document.getElementById("add-sample-modal").classList.remove("hidden");
    document.getElementById("sample-question").value = "";
    document.getElementById("sample-ground-truth").value = "";
    document.getElementById("sample-question").focus();
}

function hideAddSampleModal() {
    document.getElementById("add-sample-modal").classList.add("hidden");
}

function confirmAddSample() {
    const q = document.getElementById("sample-question").value.trim();
    const gt = document.getElementById("sample-ground-truth").value.trim();
    if (!q) return showToast("请输入评估问题", "error");

    state.evalDataset.push({ question: q, ground_truth: gt });
    renderDatasetTable();
    updateEvalButtons();
    hideAddSampleModal();
    showToast("已添加样本", "success");
}

// ==================== 单条评估 ====================

async function runSingleEval(index) {
    if (!state.currentKB) return showToast("请先选择知识库", "error");
    const sample = state.evalDataset[index];
    if (!sample) return;
    console.log('sample',sample)

    state.evalSingleResults[index] = { status: "running" };
    renderDatasetTable();

    try {
        const res = await fetch(`${API}/evaluations`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: sample.question,
                knowledge_base_id: state.currentKB.id,
                ground_truth: sample.ground_truth,
                supporting_titles:sample.metadata.supporting_titles,
            }),
        });
        const json = await res.json();

        if (json.code === 200 && json.data) {
            state.evalSingleResults[index] = { status: "done", data: json.data };
            showToast(`样本 #${index + 1} 评估完成`, "success");
        } else {
            state.evalSingleResults[index] = { status: "error", error: json.message || "评估失败" };
            showToast(`样本 #${index + 1} 评估失败: ${json.message || "未知错误"}`, "error");
        }
    } catch (e) {
        state.evalSingleResults[index] = { status: "error", error: e.message };
        showToast(`样本 #${index + 1} 评估失败: ${e.message}`, "error");
    }

    renderDatasetTable();
}

function showEvalDetail(index) {
    const result = state.evalSingleResults[index];
    if (!result || result.status !== "done") return;

    const data = result.data;
    const sample = state.evalDataset[index];
    const scores = data.scores;

    // RAGAS 指标卡片
    const metricsConfig = [
        { label: "忠实度", value: scores.faithfulness },
        { label: "相关性", value: scores.answer_relevancy },
        { label: "召回率", value: scores.context_recall },
        { label: "正确性", value: scores.answer_correctness },
        { label: "综合", value: scores.overall },
    ];
    document.getElementById("detail-scores").innerHTML = metricsConfig.map(m => {
        const tier = getScoreTier(m.value);
        return `
            <div class="metric-card ${tier}" style="padding: 10px 8px;">
                <div class="metric-label" style="font-size: 0.625rem;">${m.label}</div>
                <div class="metric-value" style="font-size: 1.125rem;">${(m.value * 100).toFixed(1)}%</div>
                <div class="metric-bar"><div class="metric-bar-fill" style="width: ${(m.value * 100).toFixed(1)}%"></div></div>
            </div>
        `;
    }).join("");

    // HotpotQA 官方指标（EM + F1），-1 表示未计算
    const hotpotMetricsEl = document.getElementById("detail-hotpot-metrics");
    const hasEM = scores.exact_match !== undefined && scores.exact_match >= 0;
    const hasF1 = scores.token_f1 !== undefined && scores.token_f1 >= 0;
    if (hasEM || hasF1) {
        hotpotMetricsEl.classList.remove("hidden");
        if (hasEM) {
            const emPct = (scores.exact_match * 100).toFixed(1);
            const emTier = getScoreTier(scores.exact_match);
            document.getElementById("detail-em-card").className = `metric-card ${emTier}`;
            document.getElementById("detail-em-value").textContent = `${emPct}%`;
            document.getElementById("detail-em-bar").style.width = `${emPct}%`;
        }
        if (hasF1) {
            const f1Pct = (scores.token_f1 * 100).toFixed(1);
            const f1Tier = getScoreTier(scores.token_f1);
            document.getElementById("detail-f1-card").className = `metric-card ${f1Tier}`;
            document.getElementById("detail-f1-value").textContent = `${f1Pct}%`;
            document.getElementById("detail-f1-bar").style.width = `${f1Pct}%`;
        }
    } else {
        hotpotMetricsEl.classList.add("hidden");
    }

    // HotpotQA 样本元信息（type/level/supporting_titles）
    const meta = sample.metadata || {};
    const hotpotMetaEl = document.getElementById("detail-hotpot-meta");
    if (meta.hotpotqa_id) {
        hotpotMetaEl.classList.remove("hidden");
        document.getElementById("detail-hq-type").textContent = meta.type || "—";
        document.getElementById("detail-hq-level").textContent = meta.level || "—";
        const supportingEl = document.getElementById("detail-hq-supporting");
        const supportingListEl = document.getElementById("detail-hq-supporting-list");
        if (meta.supporting_titles?.length) {
            supportingEl.classList.remove("hidden");
            supportingListEl.innerHTML = meta.supporting_titles.map(t =>
                `<span class="px-2 py-0.5 bg-violet-900/40 border border-violet-700/40 rounded text-[10px] text-violet-300">${escapeHtml(t)}</span>`
            ).join("");
        } else {
            supportingEl.classList.add("hidden");
        }
    } else {
        hotpotMetaEl.classList.add("hidden");
    }

    document.getElementById("detail-question").textContent = data.question;
    document.getElementById("detail-answer").innerHTML = renderMarkdown(data.answer);

    const gtSection = document.getElementById("detail-gt-section");
    if (sample.ground_truth) {
        gtSection.classList.remove("hidden");
        document.getElementById("detail-ground-truth").textContent = sample.ground_truth;
    } else {
        gtSection.classList.add("hidden");
    }

    const ctxSection = document.getElementById("detail-ctx-section");
    const ctxContainer = document.getElementById("detail-contexts");
    if (data.contexts && data.contexts.length > 0) {
        ctxSection.classList.remove("hidden");
        // 高亮支撑文章（若有 supporting_titles 元信息）
        const supportingTitles = meta.supporting_titles || [];
        ctxContainer.innerHTML = data.contexts.map((ctx, i) => {
            // 尝试从 context 文本开头提取标题（格式：标题\n内容）
            const lines = ctx.split("\n");
            const title = lines[0] || `片段 ${i + 1}`;
            const isSupporting = supportingTitles.some(t => ctx.includes(t) || title.includes(t));
            const borderClass = isSupporting ? "border-violet-600/50" : "border-gray-700/50";
            const titleBadge = isSupporting
                ? `<span class="ml-2 px-1 py-0.5 bg-violet-900/50 text-violet-300 text-[9px] rounded">支撑文章</span>`
                : "";
            return `
                <div class="bg-gray-800/70 border ${borderClass} rounded-lg p-3 text-xs text-gray-400">
                    <div class="flex items-center mb-1">
                        <span class="text-gray-500 font-medium">片段 ${i + 1}</span>${titleBadge}
                    </div>
                    <p class="whitespace-pre-wrap">${escapeHtml(ctx)}</p>
                </div>
            `;
        }).join("");
    } else {
        ctxSection.classList.add("hidden");
    }

    document.getElementById("eval-detail-modal").classList.remove("hidden");
}

function hideEvalDetailModal() {
    document.getElementById("eval-detail-modal").classList.add("hidden");
}

// ==================== 批量评估 ====================

async function runBatchEval() {
    if (!state.currentKB || state.evalDataset.length === 0 || state.isEvaluating) return;

    state.isEvaluating = true;
    updateEvalButtons();

    const resultsSection = document.getElementById("eval-results-section");
    resultsSection.classList.remove("hidden");
    document.getElementById("eval-result-info").textContent = "正在评估中...";
    document.getElementById("eval-summary").innerHTML = buildProgressCard();
    document.getElementById("eval-results-tbody").innerHTML = "";

    showToast(`开始评估 ${state.evalDataset.length} 条样本...`, "info");

    try {
        const res = await fetch(`${API}/evaluations/batch`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                knowledge_base_id: state.currentKB.id,
                samples: state.evalDataset,
            }),
        });
        const json = await res.json();

        if (json.code === 200 && json.data) {
            state.evalResults = json.data;
            renderEvalResults();
            showToast("评估完成！", "success");
        } else {
            showToast(json.message || "评估失败", "error");
            document.getElementById("eval-result-info").textContent = "评估失败";
            document.getElementById("eval-summary").innerHTML = "";
        }
    } catch (e) {
        showToast("评估失败: " + e.message, "error");
        document.getElementById("eval-result-info").textContent = "评估失败: " + e.message;
        document.getElementById("eval-summary").innerHTML = "";
    }

    state.isEvaluating = false;
    updateEvalButtons();
}

function buildProgressCard() {
    return `
        <div class="col-span-full flex items-center justify-center gap-3 py-6">
            <div class="eval-spinner"></div>
            <span class="text-sm text-gray-400">正在逐条运行 RAG + RAGAS 评估，请耐心等待...</span>
        </div>
    `;
}

// ==================== 评估结果渲染 ====================

function renderEvalResults() {
    if (!state.evalResults) return;

    const { summary, results } = state.evalResults;
    document.getElementById("eval-result-info").textContent = `共评估 ${summary.total_samples} 条样本`;

    const metrics = [
        { label: "忠实度", key: "avg_faithfulness", icon: "F" },
        { label: "相关性", key: "avg_answer_relevancy", icon: "R" },
        { label: "召回率", key: "avg_context_recall", icon: "C" },
        { label: "正确性", key: "avg_answer_correctness", icon: "A" },
    ];

    document.getElementById("eval-summary").innerHTML = metrics.map(m => {
        const val = summary[m.key] || 0;
        const tier = getScoreTier(val);
        return `
            <div class="metric-card ${tier}">
                <div class="metric-label">${m.label}</div>
                <div class="metric-value">${(val * 100).toFixed(1)}%</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: ${(val * 100).toFixed(1)}%"></div>
                </div>
            </div>
        `;
    }).join("");

    const tbody = document.getElementById("eval-results-tbody");
    tbody.innerHTML = results.map((r, i) => `
        <tr>
            <td class="text-gray-500">${i + 1}</td>
            <td class="max-w-[200px]">
                <div class="truncate" title="${escapeHtml(r.question)}">${escapeHtml(r.question)}</div>
            </td>
            <td class="max-w-[240px]">
                <div class="truncate text-gray-400" title="${escapeHtml(r.answer)}">${escapeHtml(r.answer)}</div>
            </td>
            <td class="text-center">${scorePill(r.scores.faithfulness)}</td>
            <td class="text-center">${scorePill(r.scores.answer_relevancy)}</td>
            <td class="text-center">${scorePill(r.scores.context_recall)}</td>
            <td class="text-center">${scorePill(r.scores.answer_correctness)}</td>
            <td class="text-center">${scorePill(r.scores.overall)}</td>
        </tr>
    `).join("");

    const resultsEl = document.getElementById("eval-results-section");
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function getScoreTier(score) {
    if (score >= 0.7) return "score-high";
    if (score >= 0.4) return "score-mid";
    return "score-low";
}

function scorePill(score) {
    const pct = (score * 100).toFixed(1);
    const tier = score >= 0.7 ? "high" : score >= 0.4 ? "mid" : "low";
    return `<span class="score-pill ${tier}">${pct}%</span>`;
}

// ==================== 监控面板 ====================

let _metricsTimer = null;

function parsePrometheusText(text) {
    const metrics = {};
    for (const line of text.split("\n")) {
        if (!line || line.startsWith("#")) continue;
        const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*?)(\{.*?\})?\s+(.+?)(\s+\d+)?$/);
        if (!match) continue;
        const name = match[1];
        const labels = match[2] || "";
        const value = parseFloat(match[3]);
        if (!metrics[name]) metrics[name] = [];
        const labelObj = {};
        if (labels) {
            for (const m of labels.slice(1, -1).matchAll(/(\w+)="([^"]*)"/g)) {
                labelObj[m[1]] = m[2];
            }
        }
        metrics[name].push({ labels: labelObj, value });
    }
    return metrics;
}

function getMetricValue(metrics, name, labelFilter) {
    const entries = metrics[name];
    if (!entries) return 0;
    if (!labelFilter) return entries[0]?.value || 0;
    const found = entries.find(e =>
        Object.entries(labelFilter).every(([k, v]) => e.labels[k] === v)
    );
    return found?.value || 0;
}

function sumMetric(metrics, name, labelFilter) {
    const entries = metrics[name];
    if (!entries) return 0;
    if (!labelFilter) return entries.reduce((s, e) => s + e.value, 0);
    return entries
        .filter(e => Object.entries(labelFilter).every(([k, v]) => e.labels[k] === v))
        .reduce((s, e) => s + e.value, 0);
}

function fmtNum(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(1) + "K";
    if (Number.isInteger(n)) return n.toString();
    return n.toFixed(2);
}

function buildMonCard(label, value, sub, color) {
    return `
        <div class="mon-card ${color}">
            <div class="mon-card-label">${label}</div>
            <div class="mon-card-value">${value}</div>
            ${sub ? `<div class="mon-card-sub">${sub}</div>` : ""}
        </div>
    `;
}

function buildHitRateCard(label, rate, hits, misses, color) {
    const pct = isNaN(rate) ? 0 : rate;
    return `
        <div class="mon-card ${color}">
            <div class="mon-card-label">${label}</div>
            <div class="mon-card-value">${pct.toFixed(1)}%</div>
            <div class="mon-hit-bar"><div class="mon-hit-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
            <div class="mon-card-sub">命中 ${fmtNum(hits)} / 未命中 ${fmtNum(misses)}</div>
        </div>
    `;
}

async function fetchAndRenderMetrics() {
    try {
        const res = await fetch(`${API}/metrics`);
        const text = await res.text();
        const m = parsePrometheusText(text);

        renderRequestMetrics(m);
        renderRAGMetrics(m);
        renderDocMetrics(m);
        renderSystemMetrics(m);

        document.getElementById("raw-metrics-content").textContent = text;

        const now = new Date();
        document.getElementById("metrics-update-time").textContent =
            `最后更新: ${now.toLocaleTimeString("zh-CN")}`;
    } catch (e) {
        document.getElementById("metrics-update-time").textContent = `加载失败: ${e.message}`;
    }
}

function renderRequestMetrics(m) {
    const totalReq = sumMetric(m, "rag_request_total");
    const totalReqCreated = sumMetric(m, "rag_request_total_created");
    const durationSum = sumMetric(m, "rag_request_duration_seconds_sum");
    const durationCount = sumMetric(m, "rag_request_duration_seconds_count");
    const avgLatency = durationCount > 0 ? (durationSum / durationCount * 1000) : 0;

    const p95Bucket = estimatePercentile(m, "rag_request_duration_seconds_bucket", 0.95);

    const statusOK = m["rag_request_total"]
        ?.filter(e => e.labels.status_code?.startsWith("2"))
        .reduce((s, e) => s + e.value, 0) || 0;
    const statusErr = totalReq - statusOK;

    document.getElementById("metrics-request").innerHTML = [
        buildMonCard("总请求数", fmtNum(totalReq), `成功 ${fmtNum(statusOK)} / 错误 ${fmtNum(statusErr)}`, "blue"),
        buildMonCard("平均延迟", avgLatency > 0 ? avgLatency.toFixed(0) + "ms" : "N/A", `共 ${fmtNum(durationCount)} 次采样`, "cyan"),
        buildMonCard("P95 延迟", p95Bucket > 0 ? (p95Bucket * 1000).toFixed(0) + "ms" : "N/A", "第95百分位估算", "purple"),
        buildMonCard("错误数", fmtNum(statusErr), statusErr > 0 ? `错误率 ${(statusErr / Math.max(totalReq, 1) * 100).toFixed(1)}%` : "无错误", statusErr > 0 ? "rose" : "emerald"),
    ].join("");
}

function renderRAGMetrics(m) {
    const cacheHit = sumMetric(m, "rag_cache_hit_total");
    const cacheMiss = sumMetric(m, "rag_cache_miss_total");
    const cacheTotal = cacheHit + cacheMiss;
    const hitRate = cacheTotal > 0 ? (cacheHit / cacheTotal * 100) : 0;
    const cacheSize = getMetricValue(m, "rag_cache_size");

    const retrievalDocsSum = sumMetric(m, "rag_retrieval_documents_sum");
    const retrievalDocsCount = sumMetric(m, "rag_retrieval_documents_count");
    const avgDocs = retrievalDocsCount > 0 ? (retrievalDocsSum / retrievalDocsCount) : 0;

    document.getElementById("metrics-rag").innerHTML = [
        buildHitRateCard("缓存命中率", hitRate, cacheHit, cacheMiss, "emerald"),
        buildMonCard("缓存大小", fmtNum(cacheSize), "当前缓存条目数", "teal"),
        buildMonCard("检索次数", fmtNum(retrievalDocsCount), `平均返回 ${avgDocs.toFixed(1)} 篇文档`, "purple"),
        buildMonCard("LLM Token", fmtNum(sumMetric(m, "rag_llm_token_total")), `Prompt + Completion`, "indigo"),
    ].join("");
}

function renderDocMetrics(m) {
    const docCount = sumMetric(m, "rag_document_processing_seconds_count");
    const docSum = sumMetric(m, "rag_document_processing_seconds_sum");
    const avgDocTime = docCount > 0 ? (docSum / docCount) : 0;

    const chunkCount = sumMetric(m, "rag_document_chunk_count_count");
    const chunkSum = sumMetric(m, "rag_document_chunk_count_sum");
    const avgChunks = chunkCount > 0 ? (chunkSum / chunkCount) : 0;

    const embedCount = sumMetric(m, "rag_embedding_batch_size_count");
    const embedSum = sumMetric(m, "rag_embedding_batch_size_sum");
    const avgBatch = embedCount > 0 ? (embedSum / embedCount) : 0;

    document.getElementById("metrics-doc").innerHTML = [
        buildMonCard("已处理文档", fmtNum(docCount), docCount > 0 ? `总耗时 ${docSum.toFixed(1)}s` : "暂无数据", "amber"),
        buildMonCard("平均处理耗时", avgDocTime > 0 ? avgDocTime.toFixed(1) + "s" : "N/A", `共 ${fmtNum(docCount)} 篇`, "blue"),
        buildMonCard("平均分块数", avgChunks > 0 ? avgChunks.toFixed(0) : "N/A", `共 ${fmtNum(chunkSum)} 块`, "purple"),
        buildMonCard("平均 Embed 批次", avgBatch > 0 ? avgBatch.toFixed(0) : "N/A", `共 ${fmtNum(embedCount)} 次`, "cyan"),
    ].join("");
}

function renderSystemMetrics(m) {
    const wsConn = getMetricValue(m, "rag_active_websocket_connections");
    const vectorSize = getMetricValue(m, "rag_vector_store_size");

    const evalEntries = m["rag_evaluation_score"] || [];
    const evalCards = [
        { label: "忠实度", key: "faithfulness", color: "emerald" },
        { label: "相关性", key: "answer_relevancy", color: "blue" },
        { label: "召回率", key: "context_recall", color: "purple" },
        { label: "正确性", key: "answer_correctness", color: "amber" },
    ];

    let html = [
        buildMonCard("WebSocket 连接", fmtNum(wsConn), "当前活跃连接数", "cyan"),
        buildMonCard("向量数量", fmtNum(vectorSize), "FAISS 索引", "indigo"),
    ];

    for (const ec of evalCards) {
        const val = evalEntries.find(e => e.labels.metric === ec.key)?.value || 0;
        const pct = (val * 100).toFixed(1);
        html.push(buildMonCard(`评估: ${ec.label}`, val > 0 ? pct + "%" : "N/A", "最近一次评估", ec.color));
    }

    document.getElementById("metrics-system").innerHTML = html.join("");
}

function estimatePercentile(metrics, bucketName, percentile) {
    const entries = metrics[bucketName];
    if (!entries || entries.length === 0) return 0;

    const grouped = {};
    for (const e of entries) {
        const key = Object.entries(e.labels)
            .filter(([k]) => k !== "le")
            .map(([k, v]) => `${k}=${v}`)
            .join(",");
        if (!grouped[key]) grouped[key] = [];
        grouped[key].push(e);
    }

    let totalCount = 0;
    let allBuckets = [];

    for (const buckets of Object.values(grouped)) {
        buckets.sort((a, b) => parseFloat(a.labels.le) - parseFloat(b.labels.le));
        for (const b of buckets) {
            const le = b.labels.le === "+Inf" ? Infinity : parseFloat(b.labels.le);
            allBuckets.push({ le, count: b.value });
        }
        const inf = buckets.find(b => b.labels.le === "+Inf");
        if (inf) totalCount += inf.value;
    }

    if (totalCount === 0) return 0;

    allBuckets.sort((a, b) => a.le - b.le);

    const mergedBuckets = [];
    for (const b of allBuckets) {
        const existing = mergedBuckets.find(mb => mb.le === b.le);
        if (existing) existing.count += b.count;
        else mergedBuckets.push({ ...b });
    }

    const target = percentile * totalCount;
    let prev = { le: 0, count: 0 };
    for (const b of mergedBuckets) {
        if (b.count >= target) {
            if (b.le === Infinity) return prev.le;
            const fraction = (target - prev.count) / Math.max(b.count - prev.count, 1);
            return prev.le + fraction * (b.le - prev.le);
        }
        prev = b;
    }
    return prev.le;
}

function toggleRawMetrics() {
    const content = document.getElementById("raw-metrics-content");
    const btn = document.getElementById("raw-metrics-toggle");
    if (content.classList.contains("hidden")) {
        content.classList.remove("hidden");
        btn.textContent = "收起";
    } else {
        content.classList.add("hidden");
        btn.textContent = "展开";
    }
}

function startMetricsAutoRefresh() {
    stopMetricsAutoRefresh();
    const checkbox = document.getElementById("metrics-auto-refresh");
    if (checkbox && checkbox.checked) {
        _metricsTimer = setInterval(fetchAndRenderMetrics, 10000);
    }
}

function stopMetricsAutoRefresh() {
    if (_metricsTimer) {
        clearInterval(_metricsTimer);
        _metricsTimer = null;
    }
}

function toggleMetricsAutoRefresh(checkbox) {
    if (checkbox.checked && state.currentView === "metrics") {
        startMetricsAutoRefresh();
    } else {
        stopMetricsAutoRefresh();
    }
}

// ==================== Toast ====================

function showToast(message, type = "info") {
    const toast = document.getElementById("toast");
    const icon = document.getElementById("toast-icon");
    const text = document.getElementById("toast-text");

    const iconMap = {
        success: '<svg class="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>',
        error: '<svg class="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>',
        info: '<svg class="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
    };

    icon.innerHTML = iconMap[type] || iconMap.info;
    text.textContent = message;
    toast.classList.remove("hidden");

    setTimeout(() => toast.classList.add("hidden"), 3000);
}


// ══════════════════════════════════════════════════════════════════════
// 评估历史记录
// ══════════════════════════════════════════════════════════════════════

/** 当前选中用于对比的 run_id 列表（最多 2 条） */
let _selectedRunIds = [];

/** 加载评估历史列表 */
async function loadEvalHistory() {
    const tbody = document.getElementById('eval-history-tbody');
    tbody.innerHTML = '<tr><td colspan="12" class="text-center py-6 text-gray-500 text-xs">加载中...</td></tr>';

    try {
        const res = await fetch('/api/v1/evaluations/history?limit=50');
        const json = await res.json();
        const runs = json.data || [];

        if (runs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="12" class="text-center py-8 text-gray-600 text-xs">暂无评估历史</td></tr>';
            return;
        }

        _selectedRunIds = [];
        _updateCompareBtn();

        tbody.innerHTML = runs.map((r, idx) => {
            const fmt = v => (v === null || v === undefined) ? '<span class="text-gray-600">—</span>' : `<span class="${_scoreColor(v)}">${(v * 100).toFixed(1)}%</span>`;
            const typeBadge = r.run_type === 'batch'
                ? '<span class="px-1.5 py-0.5 bg-blue-900/40 text-blue-400 rounded text-xs">批量</span>'
                : '<span class="px-1.5 py-0.5 bg-violet-900/40 text-violet-400 rounded text-xs">单条</span>';
            const timeStr = r.created_at ? new Date(r.created_at).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '';
            return `<tr class="border-t border-gray-800 hover:bg-gray-800/40 transition cursor-pointer" data-run-id="${r.run_id}">
                <td class="px-3 py-2 text-center">
                    <input type="checkbox" class="history-check accent-emerald-500 cursor-pointer" data-run-id="${r.run_id}" onchange="toggleRunSelect(this)" onclick="event.stopPropagation()">
                </td>
                <td class="px-3 py-2 text-gray-200 max-w-[160px] truncate" title="${r.name}">${r.name || '—'}</td>
                <td class="px-3 py-2">${typeBadge}</td>
                <td class="px-3 py-2 text-center text-gray-300">${r.total_samples}</td>
                <td class="px-3 py-2 text-center">${fmt(r.avg_faithfulness)}</td>
                <td class="px-3 py-2 text-center">${fmt(r.avg_answer_relevancy)}</td>
                <td class="px-3 py-2 text-center">${fmt(r.avg_context_recall)}</td>
                <td class="px-3 py-2 text-center">${fmt(r.avg_answer_correctness)}</td>
                <td class="px-3 py-2 text-center">${fmt(r.avg_exact_match)}</td>
                <td class="px-3 py-2 text-center">${fmt(r.avg_token_f1)}</td>
                <td class="px-3 py-2 text-gray-500">${timeStr}</td>
                <td class="px-3 py-2 text-center">
                    <button class="text-gray-500 hover:text-red-400 transition" title="删除" onclick="deleteEvalRun('${r.run_id}', event)">
                        <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
                    </button>
                </td>
            </tr>`;
        }).join('');

        // 行点击展开详情
        tbody.querySelectorAll('tr[data-run-id]').forEach(tr => {
            tr.addEventListener('click', () => showEvalRunDetail(tr.dataset.runId));
        });

    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="12" class="text-center py-6 text-red-400 text-xs">加载失败：${e.message}</td></tr>`;
    }
}

/** 根据分数返回颜色 class */
function _scoreColor(v) {
    if (v >= 0.7) return 'text-emerald-400 font-medium';
    if (v >= 0.4) return 'text-yellow-400';
    return 'text-red-400';
}

/** 勾选/取消勾选 run，维护对比列表 */
function toggleRunSelect(checkbox) {
    const runId = checkbox.dataset.runId;
    if (checkbox.checked) {
        if (_selectedRunIds.length >= 2) {
            checkbox.checked = false;
            showToast('最多选择 2 条进行对比', 'warning');
            return;
        }
        _selectedRunIds.push(runId);
    } else {
        _selectedRunIds = _selectedRunIds.filter(id => id !== runId);
    }
    _updateCompareBtn();
}

function _updateCompareBtn() {
    const btn = document.getElementById('compare-btn');
    if (!btn) return;
    if (_selectedRunIds.length === 2) {
        btn.classList.remove('hidden');
        btn.disabled = false;
    } else if (_selectedRunIds.length < 2) {
        btn.classList.toggle('hidden', _selectedRunIds.length === 0);
        btn.disabled = true;
    }
}

/** 展示单次评估详情 Modal */
async function showEvalRunDetail(runId) {
    const modal = document.getElementById('eval-history-modal');
    const body = document.getElementById('history-modal-body');
    const title = document.getElementById('history-modal-title');
    const subtitle = document.getElementById('history-modal-subtitle');

    body.innerHTML = '<div class="text-center py-10 text-gray-500 text-xs">加载中...</div>';
    modal.classList.remove('hidden');

    try {
        const res = await fetch(`/api/v1/evaluations/history/${runId}`);
        const json = await res.json();
        const d = json.data;

        title.textContent = d.name || '评估详情';
        subtitle.textContent = `${d.run_type === 'batch' ? '批量' : '单条'} · ${d.total_samples} 条样本 · ${new Date(d.created_at).toLocaleString('zh-CN')}`;

        const fmt = v => (v === null || v === undefined) ? '—' : `<span class="${_scoreColor(v)}">${(v * 100).toFixed(1)}%</span>`;

        // 汇总指标
        const summaryHtml = `
        <div class="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-5">
            ${[
                ['忠实度', d.avg_faithfulness],
                ['相关性', d.avg_answer_relevancy],
                ['召回率', d.avg_context_recall],
                ['正确性', d.avg_answer_correctness],
                ['EM', d.avg_exact_match],
                ['F1', d.avg_token_f1],
            ].map(([label, val]) => `
            <div class="bg-gray-800 rounded-lg p-2.5 text-center">
                <div class="text-xs text-gray-500 mb-1">${label}</div>
                <div class="text-sm font-semibold">${fmt(val)}</div>
            </div>`).join('')}
        </div>`;

        // 明细表格
        const rowsHtml = d.items.map((it, i) => `
        <tr class="border-t border-gray-800 hover:bg-gray-800/30 transition">
            <td class="px-3 py-2 text-gray-500">${i + 1}</td>
            <td class="px-3 py-2 text-gray-200 max-w-[200px]">
                <div class="truncate" title="${it.question}">${it.question}</div>
                ${it.ground_truth ? `<div class="text-gray-500 truncate mt-0.5" title="${it.ground_truth}">标准：${it.ground_truth}</div>` : ''}
            </td>
            <td class="px-3 py-2 text-gray-300 max-w-[200px]">
                <div class="truncate" title="${it.answer}">${it.answer || '—'}</div>
            </td>
            <td class="px-3 py-2 text-center text-xs">${fmt(it.faithfulness)}</td>
            <td class="px-3 py-2 text-center text-xs">${fmt(it.answer_relevancy)}</td>
            <td class="px-3 py-2 text-center text-xs">${fmt(it.context_recall)}</td>
            <td class="px-3 py-2 text-center text-xs">${fmt(it.answer_correctness)}</td>
            <td class="px-3 py-2 text-center text-xs">${fmt(it.overall_score)}</td>
            <td class="px-3 py-2 text-center text-xs">${fmt(it.exact_match)}</td>
            <td class="px-3 py-2 text-center text-xs">${fmt(it.token_f1)}</td>
        </tr>`).join('');

        body.innerHTML = summaryHtml + `
        <div class="overflow-x-auto rounded-lg border border-gray-800 max-h-[50vh] overflow-y-auto">
            <table class="w-full text-xs">
                <thead class="bg-gray-800/80 sticky top-0">
                    <tr>
                        <th class="text-left px-3 py-2 text-gray-400 w-8">#</th>
                        <th class="text-left px-3 py-2 text-gray-400">问题 / 标准答案</th>
                        <th class="text-left px-3 py-2 text-gray-400">RAG 回答</th>
                        <th class="text-center px-3 py-2 text-gray-400 w-14">忠实度</th>
                        <th class="text-center px-3 py-2 text-gray-400 w-14">相关性</th>
                        <th class="text-center px-3 py-2 text-gray-400 w-14">召回率</th>
                        <th class="text-center px-3 py-2 text-gray-400 w-14">正确性</th>
                        <th class="text-center px-3 py-2 text-gray-400 w-14">综合</th>
                        <th class="text-center px-3 py-2 text-gray-400 w-10">EM</th>
                        <th class="text-center px-3 py-2 text-gray-400 w-10">F1</th>
                    </tr>
                </thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>`;
    } catch (e) {
        body.innerHTML = `<div class="text-center py-10 text-red-400 text-xs">加载失败：${e.message}</div>`;
    }
}

function closeHistoryModal() {
    document.getElementById('eval-history-modal').classList.add('hidden');
}

/** 对比两条选中的 EvalRun */
async function compareSelectedRuns() {
    if (_selectedRunIds.length !== 2) return;
    const [a, b] = _selectedRunIds;
    const modal = document.getElementById('eval-compare-modal');
    const body = document.getElementById('compare-modal-body');

    body.innerHTML = '<div class="text-center py-10 text-gray-500 text-xs">加载中...</div>';
    modal.classList.remove('hidden');

    try {
        const res = await fetch(`/api/v1/evaluations/history/compare/${a}/${b}`);
        const json = await res.json();
        const { run_a, run_b, delta } = json.data;

        const fmt = v => (v === null || v === undefined) ? '—' : `${(v * 100).toFixed(1)}%`;
        const fmtDelta = v => {
            if (v === null || v === undefined) return '<span class="text-gray-600">—</span>';
            const sign = v > 0 ? '+' : '';
            const cls = v > 0.005 ? 'text-emerald-400' : v < -0.005 ? 'text-red-400' : 'text-gray-400';
            return `<span class="${cls}">${sign}${(v * 100).toFixed(1)}%</span>`;
        };

        const metrics = [
            ['忠实度', 'avg_faithfulness'],
            ['答案相关性', 'avg_answer_relevancy'],
            ['上下文召回', 'avg_context_recall'],
            ['答案正确性', 'avg_answer_correctness'],
            ['Exact Match', 'avg_exact_match'],
            ['Token F1', 'avg_token_f1'],
        ];

        body.innerHTML = `
        <div class="grid grid-cols-3 gap-3 mb-4 text-xs text-center">
            <div class="bg-gray-800 rounded-lg p-3">
                <div class="text-gray-500 mb-1">Run A</div>
                <div class="text-white font-medium truncate" title="${run_a.name}">${run_a.name || run_a.run_id.slice(0, 8)}</div>
                <div class="text-gray-600 mt-0.5">${new Date(run_a.created_at).toLocaleDateString('zh-CN')}</div>
            </div>
            <div class="flex items-center justify-center text-gray-500 font-bold text-lg">VS</div>
            <div class="bg-gray-800 rounded-lg p-3">
                <div class="text-gray-500 mb-1">Run B</div>
                <div class="text-white font-medium truncate" title="${run_b.name}">${run_b.name || run_b.run_id.slice(0, 8)}</div>
                <div class="text-gray-600 mt-0.5">${new Date(run_b.created_at).toLocaleDateString('zh-CN')}</div>
            </div>
        </div>
        <div class="rounded-lg border border-gray-800 overflow-hidden">
            <table class="w-full text-xs">
                <thead class="bg-gray-800/80">
                    <tr>
                        <th class="text-left px-4 py-2.5 text-gray-400 font-medium">指标</th>
                        <th class="text-center px-4 py-2.5 text-gray-400 font-medium">Run A</th>
                        <th class="text-center px-4 py-2.5 text-gray-400 font-medium">Run B</th>
                        <th class="text-center px-4 py-2.5 text-gray-400 font-medium">差值 (B−A)</th>
                    </tr>
                </thead>
                <tbody>
                    ${metrics.map(([label, key]) => `
                    <tr class="border-t border-gray-800">
                        <td class="px-4 py-2.5 text-gray-300">${label}</td>
                        <td class="px-4 py-2.5 text-center ${_scoreColor(run_a[key])}">${fmt(run_a[key])}</td>
                        <td class="px-4 py-2.5 text-center ${_scoreColor(run_b[key])}">${fmt(run_b[key])}</td>
                        <td class="px-4 py-2.5 text-center">${fmtDelta(delta[key])}</td>
                    </tr>`).join('')}
                </tbody>
            </table>
        </div>`;
    } catch (e) {
        body.innerHTML = `<div class="text-center py-10 text-red-400 text-xs">加载失败：${e.message}</div>`;
    }
}

function closeCompareModal() {
    document.getElementById('eval-compare-modal').classList.add('hidden');
}

/** 删除一条评估历史 */
async function deleteEvalRun(runId, event) {
    event.stopPropagation();
    if (!confirm('确认删除这条评估记录？此操作不可恢复。')) return;
    try {
        const res = await fetch(`/api/v1/evaluations/history/${runId}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(await res.text());
        showToast('已删除', 'success');
        loadEvalHistory();
    } catch (e) {
        showToast(`删除失败：${e.message}`, 'error');
    }
}
