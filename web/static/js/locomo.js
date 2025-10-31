// Locomo benchmark tab functionality

let locomoData = null;

window.loadLocomoResults = async function() {
    try {
        const response = await fetch('/api/locomo');
        locomoData = await response.json();
        renderLocomoResults();
    } catch (e) {
        document.getElementById('locomo-content').innerHTML = `
            <div class="error-message">Error loading benchmark results: ${e.message}</div>
        `;
    }
}

function renderLocomoResults() {
    if (!locomoData) return;

    const content = document.getElementById('locomo-content');

    // Overall stats
    const overallHtml = `
        <div style="background: #f9f9f9; padding: 20px; border: 2px solid #333; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin-top: 0;">Overall Performance</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">Overall Accuracy</div>
                    <div class="stat-value">${locomoData.overall_accuracy.toFixed(2)}%</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Correct Answers</div>
                    <div class="stat-value">${locomoData.total_correct} / ${locomoData.total_questions}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Conversations</div>
                    <div class="stat-value">${locomoData.conversation_results.length}</div>
                </div>
            </div>
        </div>
    `;

    // Filter controls
    const filterHtml = `
        <div style="margin-bottom: 20px; display: flex; gap: 10px; align-items: center;">
            <label style="font-weight: bold;">Show:</label>
            <label><input type="radio" name="answer-filter" value="all" checked onchange="filterAnswers()"> All Answers</label>
            <label><input type="radio" name="answer-filter" value="incorrect" onchange="filterAnswers()"> ❌ Incorrect Only</label>
            <label><input type="radio" name="answer-filter" value="correct" onchange="filterAnswers()"> ✅ Correct Only</label>
        </div>
    `;

    // Build conversation sections
    let conversationsHtml = '';
    locomoData.conversation_results.forEach((conv, idx) => {
        const accuracy = conv.metrics.accuracy.toFixed(2);
        const correctCount = conv.metrics.correct;
        const totalCount = conv.metrics.total;

        conversationsHtml += `
            <div style="margin-bottom: 30px; border: 2px solid #333; border-radius: 8px; overflow: hidden;">
                <div style="background: #f0f0f0; padding: 15px; border-bottom: 2px solid #333; cursor: pointer;" onclick="toggleConversation(${idx})">
                    <h3 style="margin: 0; display: flex; justify-content: space-between; align-items: center;">
                        <span>📊 ${conv.sample_id}</span>
                        <span style="font-size: 18px; color: ${accuracy >= 70 ? '#43a047' : accuracy >= 50 ? '#ff9800' : '#e53935'};">
                            ${accuracy}% (${correctCount}/${totalCount})
                        </span>
                    </h3>
                </div>
                <div id="conv-${idx}" style="display: none; padding: 20px;">
                    ${renderConversationDetails(conv)}
                </div>
            </div>
        `;
    });

    content.innerHTML = overallHtml + filterHtml + conversationsHtml;
}

function renderConversationDetails(conv) {
    const results = conv.metrics.detailed_results;

    let html = '<div class="qa-results">';

    results.forEach((result, idx) => {
        const isCorrect = result.is_correct;
        const bgColor = isCorrect ? '#e8f5e9' : '#ffebee';
        const icon = isCorrect ? '✅' : '❌';
        const category = getCategoryName(result.category);

        html += `
            <div class="qa-item" data-correct="${isCorrect}" style="background: ${bgColor}; padding: 15px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                    <div style="flex: 1;">
                        <div style="font-weight: bold; font-size: 16px; margin-bottom: 8px;">
                            ${icon} Question ${idx + 1} <span style="font-size: 12px; background: #666; color: white; padding: 2px 8px; border-radius: 4px; margin-left: 8px;">${category}</span>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <b>Q:</b> ${result.question}
                        </div>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 10px;">
                    <div>
                        <div style="font-weight: bold; color: #43a047; margin-bottom: 4px;">✓ Correct Answer:</div>
                        <div style="background: white; padding: 8px; border-radius: 4px; border: 1px solid #ccc;">
                            ${result.correct_answer}
                        </div>
                    </div>
                    <div>
                        <div style="font-weight: bold; color: ${isCorrect ? '#43a047' : '#e53935'}; margin-bottom: 4px;">
                            ${isCorrect ? '✓' : '✗'} Predicted Answer:
                        </div>
                        <div style="background: white; padding: 8px; border-radius: 4px; border: 1px solid #ccc;">
                            ${result.predicted_answer}
                        </div>
                    </div>
                </div>

                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer; font-weight: bold; padding: 5px; background: rgba(255,255,255,0.5); border-radius: 4px;">
                        📝 Show Reasoning & Retrieved Memories
                    </summary>
                    <div style="margin-top: 10px; padding: 10px; background: white; border-radius: 4px;">
                        <div style="margin-bottom: 10px;">
                            <b>System Reasoning:</b>
                            <div style="padding: 8px; background: #f5f5f5; border-radius: 4px; margin-top: 4px;">
                                ${result.reasoning}
                            </div>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <b>Judge Reasoning:</b>
                            <div style="padding: 8px; background: #f5f5f5; border-radius: 4px; margin-top: 4px;">
                                ${result.correctness_reasoning || 'N/A'}
                            </div>
                        </div>
                        <div>
                            <b>Retrieved Memories (${result.retrieved_memories ? result.retrieved_memories.length : 0}):</b>
                            ${renderRetrievedMemories(result.retrieved_memories)}
                        </div>
                    </div>
                </details>
            </div>
        `;
    });

    html += '</div>';
    return html;
}

function renderRetrievedMemories(memories) {
    if (!memories || memories.length === 0) {
        return '<div style="padding: 8px; color: #999;">No memories retrieved</div>';
    }

    let html = '<div style="margin-top: 8px;">';
    memories.forEach((mem, idx) => {
        html += `
            <div style="padding: 8px; background: #f5f5f5; border-left: 3px solid #42a5f5; margin-bottom: 8px;">
                <div style="font-size: 11px; color: #666; margin-bottom: 4px;">
                    Rank #${idx + 1} | Score: ${mem.score ? mem.score.toFixed(4) : 'N/A'}
                </div>
                <div style="font-size: 13px;">${mem.text}</div>
            </div>
        `;
    });
    html += '</div>';
    return html;
}

function getCategoryName(category) {
    const categories = {
        1: 'Multi-hop',
        2: 'Single-hop',
        3: 'Temporal',
        4: 'Open-domain'
    };
    return categories[category] || 'Unknown';
}

function toggleConversation(idx) {
    const elem = document.getElementById(`conv-${idx}`);
    if (elem.style.display === 'none') {
        elem.style.display = 'block';
    } else {
        elem.style.display = 'none';
    }
}

function filterAnswers() {
    const filter = document.querySelector('input[name="answer-filter"]:checked').value;
    const items = document.querySelectorAll('.qa-item');

    items.forEach(item => {
        const isCorrect = item.dataset.correct === 'true';

        if (filter === 'all') {
            item.style.display = 'block';
        } else if (filter === 'correct' && isCorrect) {
            item.style.display = 'block';
        } else if (filter === 'incorrect' && !isCorrect) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}
