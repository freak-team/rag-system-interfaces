document.addEventListener('DOMContentLoaded', () => {
    
    const USE_MOCK = false; 
    
    const API_URLS = {
        search: '/api/search',   
        checkAnswer: '/api/check',
        getQuestion: '/api/question' 
    };

    let currentQuestionId = null; 
    const tabButtons = document.querySelectorAll('.tab-btn');
    const viewSections = document.querySelectorAll('.view-section');

    let recentQuestions = [];
    const HISTORY_LIMIT = 20;

    async function fetchNextQuestion() {
        const excludeParam = recentQuestions.join(',');
        
        const response = await fetch(`/api/question?exclude=${excludeParam}`);
        const data = await response.json();
        
        if (data.id) {
            recentQuestions.push(data.id);
            
            if (recentQuestions.length > HISTORY_LIMIT) {
                recentQuestions.shift();
            }
            
            console.log("Получен вопрос:", data.question);
        }
    }

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            const targetId = button.getAttribute('data-target');

            viewSections.forEach(section => {
                if (section.id === targetId) {
                    section.classList.remove('hidden');
                } else {
                    section.classList.add('hidden');
                }
            });

            if (targetId === 'trainer-mode' && !currentQuestionId) {
                loadQuestion();
            }
        });
    });

    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const answerArea = document.getElementById('answer-area');
    const answerText = document.getElementById('answer-text');

    async function handleSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        searchBtn.disabled = true;
        answerArea.classList.remove('hidden');
        answerText.innerHTML = '<span style="color: var(--text-muted);">Ищем ответ в материалах...</span>';

        try {
            let responseData;

            if (USE_MOCK) {
                await new Promise(resolve => setTimeout(resolve, 1500));
                responseData = {
                    answer: `<strong>Ответ системы:</strong> Вы спросили про "${query}". <br><br> В контексте дискретной математики это важное понятие. Формальное определение гласит, что это минимальное количество цветов, в которые можно раскрасить вершины графа так, чтобы концы любого ребра имели разные цвета.`
                };
            } else {
                const response = await fetch(API_URLS.search, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: query })
                });
                if (!response.ok) {
                    throw new Error(`Backend returned status ${response.status}`);
                }
                responseData = await response.json();
            }

            answerText.innerHTML = responseData.answer;

        } catch (error) {
            console.error("Ошибка поиска:", error);
            answerText.innerHTML = '<span style="color: var(--error-text);">Произошла ошибка при связи с сервером. Попробуйте позже.</span>';
        } finally {
            searchBtn.disabled = false;
        }
    }

    searchBtn.addEventListener('click', handleSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });

    const trainerQuestionElement = document.getElementById('trainer-question');
    
    async function loadQuestion() {
        trainerQuestionElement.textContent = "Загружаем вопрос...";
        feedbackArea.classList.add('hidden');
        trainerInput.value = ''; 

        try {
            let data;
            if (USE_MOCK) {
                await new Promise(resolve => setTimeout(resolve, 500));
                data = {
                    question_id: 42,
                    text: "Что такое двудольный граф?"
                };
            } else {
                const response = await fetch(API_URLS.getQuestion);
                if (!response.ok) {
                    throw new Error(`Backend returned status ${response.status}`);
                }
                data = await response.json();
            }
            
            currentQuestionId = data.question_id ?? data.id ?? null;
            trainerQuestionElement.textContent = data.text ?? data.question ?? "Не удалось получить текст вопроса";

            if (currentQuestionId === null) {
                throw new Error('Question id is missing in backend response');
            }
        } catch (error) {
            console.error("Ошибка при загрузке вопроса:", error);
            trainerQuestionElement.textContent = "Не удалось загрузить вопрос. Попробуйте обновить страницу.";
        }
    }

    const trainerInput = document.getElementById('trainer-input');
    const checkBtn = document.getElementById('check-btn');
    const feedbackArea = document.getElementById('feedback-area');
    const feedbackTitle = document.getElementById('feedback-title');
    const feedbackText = document.getElementById('feedback-text');

    async function handleCheckAnswer() {
        const studentAnswer = trainerInput.value.trim();
        if (!studentAnswer || currentQuestionId === null) return;

        checkBtn.disabled = true;
        feedbackArea.classList.add('hidden'); 

        try {
            let result;

            if (USE_MOCK) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                const isCorrect = studentAnswer.length > 10; 
                
                result = {
                    isCorrect: isCorrect,
                    explanation: isCorrect 
                        ? "Отлично! Вы верно сформулировали определение." 
                        : "Не совсем. Из учебника: «Граф называется двудольным, если его вершины можно разбить на два непересекающихся множества...»"
                };
            } else {
                const response = await fetch(API_URLS.checkAnswer, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        question_id: currentQuestionId, 
                        answer: studentAnswer 
                    })
                });
                if (!response.ok) {
                    throw new Error(`Backend returned status ${response.status}`);
                }
                result = await response.json();
            }

            feedbackArea.classList.remove('hidden', 'success', 'error');
            
            if (result.isCorrect) {
                feedbackArea.classList.add('success');
                feedbackTitle.textContent = "Верно!";
            } else {
                feedbackArea.classList.add('error');
                feedbackTitle.textContent = "Нужно подучить";
            }
            
            feedbackText.textContent = result.explanation;

        } catch (error) {
            console.error("Ошибка проверки:", error);
        } finally {
            checkBtn.disabled = false;
        }
    }

    checkBtn.addEventListener('click', handleCheckAnswer);
    const nextQuestionBtn = document.getElementById('next-question-btn');
    nextQuestionBtn.addEventListener('click', loadQuestion)
});