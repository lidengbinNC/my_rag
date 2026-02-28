/**
 * MyRAG 前端应用
 */

const API = "/api/v1";

const state = {
    currentKB: null,
    conversationId: null,
    isStreaming: false,
    knowledgeBases: [],
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

    const input = document.getElementById("chat-input");
    input.disabled = false;
    input.placeholder = `向「${state.currentKB.name}」提问...`;
    document.getElementById("send-btn").disabled = false;

    renderKBList();
    loadDocuments();
    clearChat();
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
                    <div class="text-xs text-gray-600">${formatFileSize(doc.file_size)} · <span class="status-${doc.status}">${statusText(doc.status)}</span></div>
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
            showToast("文档上传成功", "success");
            await loadDocuments();
            await loadKnowledgeBases();
        } else {
            showToast(json.message || "上传失败", "error");
        }
    } catch (e) {
        showToast("上传失败: " + e.message, "error");
    }
    input.value = "";
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

// ==================== 对话 ====================

async function sendMessage() {
    if (state.isStreaming || !state.currentKB) return;

    const input = document.getElementById("chat-input");
    const query = input.value.trim();
    if (!query) return;

    input.value = "";
    autoResize(input);

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
                stream: true,
            }),
        });

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
    } catch (e) {
        contentEl.innerHTML = `<span class="text-red-400">请求失败: ${escapeHtml(e.message)}</span>`;
    }

    state.isStreaming = false;
    updateSendButton();
    scrollToBottom();
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
