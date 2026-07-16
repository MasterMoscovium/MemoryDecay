// Client-Side Reinforcement Learning Maze Solver Sandbox
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const trainBtn = document.getElementById('trainBtn');
    const resetBtn = document.getElementById('resetBtn');
    const shutdownBtn = document.getElementById('shutdownBtn');
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    
    const lrSlider = document.getElementById('lrSlider');
    const lrVal = document.getElementById('lrVal');
    const gammaSlider = document.getElementById('gammaSlider');
    const gammaVal = document.getElementById('gammaVal');
    const episodesSlider = document.getElementById('episodesSlider');
    const episodesVal = document.getElementById('episodesVal');
    const speedSlider = document.getElementById('speedSlider');
    const speedVal = document.getElementById('speedVal');
    
    const mazeContainer = document.getElementById('maze');
    const policyMazeContainer = document.getElementById('policyMaze');
    const simTitle = document.getElementById('simTitle');
    const collisionFlash = document.getElementById('collisionFlash');
    
    const metricDecay = document.getElementById('metricDecay');
    const metricIteration = document.getElementById('metricIteration');
    const metricStep = document.getElementById('metricStep');
    const metricReward = document.getElementById('metricReward');
    const consoleLog = document.getElementById('consoleLog');
    
    // Sliders event listeners
    lrSlider.addEventListener('input', (e) => lrVal.textContent = e.target.value);
    gammaSlider.addEventListener('input', (e) => gammaVal.textContent = e.target.value);
    episodesSlider.addEventListener('input', (e) => episodesVal.textContent = e.target.value);
    speedSlider.addEventListener('input', (e) => speedVal.textContent = e.target.value + 'ms');

    // Tab Switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            const activeTabId = btn.getAttribute('data-tab');
            document.getElementById(activeTabId).classList.add('active');
            
            // Re-render grids if tabs changed to refresh view
            if (activeTabId === 'policyTab') {
                renderPolicyGrid();
            }
        });
    });

    // Static Maze Definition (More open to allow 7-8 ways to reach the goal)
    const STATIC_MAZE = [
        ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
        ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0'],
        ['0', '1', '0', '1', '0', '0', '0', '1', '0', '1', '0'],
        ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0'],
        ['0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0'],
        ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0'],
        ['0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0'],
        ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0'],
        ['0', '1', '0', '1', '0', '0', '0', '1', '0', '1', '0'],
        ['0', '1', '1', '1', '1', '1', '1', '1', '1', '2', '0'],
        ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
    ];

    // Helper functions
    function _actionTarget(state, action) {
        let [r, c] = state;
        if (action === 0) r -= 1;      // Up
        else if (action === 1) c += 1; // Right
        else if (action === 2) r += 1; // Down
        else if (action === 3) c -= 1; // Left
        return [r, c];
    }

    function arrayEquals(a, b) {
        return a[0] === b[0] && a[1] === b[1];
    }

    // Console Logging Utility
    function log(message, type = 'system') {
        const p = document.createElement('p');
        p.className = type;
        p.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
        consoleLog.appendChild(p);
        consoleLog.scrollTop = consoleLog.scrollHeight;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Environment Class
    // ────────────────────────────────────────────────────────────────────────
    class MazeEnv {
        constructor() {
            this.staticMaze = JSON.parse(JSON.stringify(STATIC_MAZE));
            this.startPos = [1, 1];
            this.targetPos = [9, 9];
            this.rows = this.staticMaze.length;
            this.cols = this.staticMaze[0].length;
            this.potentialDynamicObstacles = [[1, 3], [1, 5], [1, 7], [3, 3], [3, 5], [3, 7], [5, 3], [5, 5], [5, 7], [7, 3], [7, 5], [7, 7]];
            this.dynamicObstacles = [];
            this.currentStep = 0;
            this.updateInterval = 5;
            this.distanceMap = {};
            this.buildDistanceMap();
            this.reset();
        }

        buildDistanceMap() {
            for (let r = 0; r < this.rows; r++) {
                for (let c = 0; c < this.cols; c++) {
                    this.distanceMap[`${r},${c}`] = Infinity;
                }
            }
            let queue = [[this.targetPos[0], this.targetPos[1], 0]];
            this.distanceMap[`${this.targetPos[0]},${this.targetPos[1]}`] = 0;
            let visited = new Set([`${this.targetPos[0]},${this.targetPos[1]}`]);

            while (queue.length > 0) {
                let [r, c, dist] = queue.shift();
                const dirs = [[-1,0], [1,0], [0,-1], [0,1]];
                for (let [dr, dc] of dirs) {
                    let nr = r + dr;
                    let nc = c + dc;
                    if (nr >= 0 && nr < this.rows && nc >= 0 && nc < this.cols && this.staticMaze[nr][nc] !== '0') {
                        let key = `${nr},${nc}`;
                        if (!visited.has(key)) {
                            visited.add(key);
                            this.distanceMap[key] = dist + 1;
                            queue.push([nr, nc, dist + 1]);
                        }
                    }
                }
            }
        }

        reset() {
            this.agentPos = [...this.startPos];
            this.currentStep = 0;
            this.updateDynamicObstacles();
            return this.getState();
        }

        updateDynamicObstacles() {
            // Select 3 to 5 potential choke points to block, leaving at least a few paths open
            let count = Math.floor(Math.random() * 3) + 3;
            let shuffled = [...this.potentialDynamicObstacles].sort(() => 0.5 - Math.random());
            this.dynamicObstacles = shuffled.slice(0, count);
        }

        getState() {
            return [...this.agentPos];
        }

        step(action) {
            let [r, c] = this.agentPos;
            let [nextR, nextC] = _actionTarget([r, c], action);

            let hitWall = false;
            if (nextR < 0 || nextR >= this.rows || nextC < 0 || nextC >= this.cols || this.staticMaze[nextR][nextC] === '0') {
                hitWall = true;
            }

            let hitDynamic = this.dynamicObstacles.some(obs => obs[0] === nextR && obs[1] === nextC);

            let reward = 0;
            let done = false;

            if (hitWall) {
                reward = -10;
                done = false;
            } else if (hitDynamic) {
                reward = -100;
                done = true;
            } else {
                let prevDist = this.distanceMap[`${r},${c}`] || Infinity;
                let nextDist = this.distanceMap[`${nextR},${nextC}`] || Infinity;
                this.agentPos = [nextR, nextC];

                if (arrayEquals(this.agentPos, this.targetPos)) {
                    reward = 100;
                    done = true;
                } else {
                    let shaping = (prevDist - nextDist) * 5;
                    reward = -1 + shaping;
                    done = false;
                }
            }

            this.currentStep += 1;
            if (this.currentStep % this.updateInterval === 0) {
                this.updateDynamicObstacles();
            }

            return [this.getState(), reward, done, { hitDynamic }];
        }

        getValidActions(state) {
            let [r, c] = state;
            let valid = [];
            // Up
            if (r > 0 && this.staticMaze[r-1][c] !== '0') valid.push(0);
            // Right
            if (c < this.cols-1 && this.staticMaze[r][c+1] !== '0') valid.push(1);
            // Down
            if (r < this.rows-1 && this.staticMaze[r+1][c] !== '0') valid.push(2);
            // Left
            if (c > 0 && this.staticMaze[r][c-1] !== '0') valid.push(3);
            return valid;
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Agent Class
    // ────────────────────────────────────────────────────────────────────────
    class QLearningAgent {
        constructor(actions = [0, 1, 2, 3], decayType = 'none') {
            this.actions = actions;
            this.decayType = decayType;
            this.qTable = {};
            this.lr = parseFloat(lrSlider.value);
            this.gamma = parseFloat(gammaSlider.value);
            this.epsilon = 1.0;
        }

        getQValues(state) {
            let key = `${state[0]},${state[1]}`;
            if (!this.qTable[key]) {
                this.qTable[key] = {};
                for (let a of this.actions) {
                    this.qTable[key][a] = Math.random() * 0.02 - 0.01;
                }
            }
            return this.qTable[key];
        }

        chooseAction(state, validActions) {
            if (!validActions || validActions.length === 0) {
                validActions = this.actions;
            }

            if (Math.random() < this.epsilon) {
                return validActions[Math.floor(Math.random() * validActions.length)];
            } else {
                let qVals = this.getQValues(state);
                let maxQ = -Infinity;
                let best = [];
                for (let a of validActions) {
                    if (qVals[a] > maxQ) {
                        maxQ = qVals[a];
                        best = [a];
                    } else if (qVals[a] === maxQ) {
                        best.push(a);
                    }
                }
                return best[Math.floor(Math.random() * best.length)];
            }
        }

        learn(state, action, reward, nextState, nextValidActions, done) {
            let qVals = this.getQValues(state);
            let maxNextQ = -Infinity;

            if (done) {
                maxNextQ = 0.0;
            } else {
                let nextQVals = this.getQValues(nextState);
                if (!nextValidActions || nextValidActions.length === 0) {
                    nextValidActions = this.actions;
                }
                for (let a of nextValidActions) {
                    if (nextQVals[a] > maxNextQ) {
                        maxNextQ = nextQVals[a];
                    }
                }
            }

            let tdTarget = reward + this.gamma * maxNextQ;
            let tdError = tdTarget - qVals[action];
            qVals[action] += this.lr * tdError;
        }

        updateEpsilon(episode, maxEpisodes) {
            if (this.decayType === 'none') {
                this.epsilon = 1.0;
            } else if (this.decayType === 'linear') {
                this.epsilon = Math.max(0.01, 1.0 - (episode / maxEpisodes));
            } else if (this.decayType === 'exponential') {
                this.epsilon = Math.max(0.01, Math.pow(0.01, episode / maxEpisodes));
            } else if (this.decayType === 'cosine') {
                let fraction = episode / maxEpisodes;
                this.epsilon = 0.01 + 0.5 * (0.99) * (1 + Math.cos(fraction * Math.PI));
            }
        }

        wipeMemory(iteration, maxIterations) {
            if (this.decayType === 'none' || iteration === 0) return;
            
            let forgetFraction = 0.0;
            if (this.decayType === 'linear') {
                forgetFraction = iteration / maxIterations;
            } else if (this.decayType === 'exponential') {
                forgetFraction = 1.0 - Math.exp(-2 * iteration / maxIterations);
            } else if (this.decayType === 'cosine') {
                let fraction = iteration / maxIterations;
                forgetFraction = 0.5 * (1 - Math.cos(fraction * Math.PI));
            }

            if (forgetFraction > 0) {
                let keys = Object.keys(this.qTable);
                let numToWipe = Math.floor(keys.length * forgetFraction);
                let shuffled = keys.sort(() => 0.5 - Math.random());
                for (let i = 0; i < numToWipe; i++) {
                    let k = shuffled[i];
                    for (let a of this.actions) {
                        this.qTable[k][a] = 0.0;
                    }
                }
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Value Iteration (Replanner)
    // ────────────────────────────────────────────────────────────────────────
    function replanQTable(agent, env, dynamicObstacles) {
        const dynSet = new Set(dynamicObstacles.map(obs => `${obs[0]},${obs[1]}`));
        
        for (let sweep = 0; sweep < 25; sweep++) {
            for (let r = 0; r < env.rows; r++) {
                for (let c = 0; c < env.cols; c++) {
                    if (env.staticMaze[r][c] === '0') continue;
                    let state = [r, c];
                    let qVals = agent.getQValues(state);
                    
                    if (r === env.targetPos[0] && c === env.targetPos[1]) {
                        for (let a of agent.actions) {
                            qVals[a] = 0.0;
                        }
                        continue;
                    }

                    let valid = env.getValidActions(state);
                    for (let action of agent.actions) {
                        if (!valid.includes(action)) {
                            qVals[action] = -999.0;
                            continue;
                        }

                        let [nr, nc] = _actionTarget(state, action);
                        if (dynSet.has(`${nr},${nc}`)) {
                            qVals[action] = -999.0;
                        } else {
                            let reward = 0;
                            let maxNextQ = -Infinity;

                            if (nr === env.targetPos[0] && nc === env.targetPos[1]) {
                                reward = 100;
                                maxNextQ = 0.0;
                            } else {
                                let prevDist = env.distanceMap[`${r},${c}`] || Infinity;
                                let nextDist = env.distanceMap[`${nr},${nc}`] || Infinity;
                                let shaping = (prevDist - nextDist) * 5;
                                reward = -1 + shaping;

                                let nextQVals = agent.getQValues([nr, nc]);
                                let nextValid = env.getValidActions([nr, nc]);
                                for (let na of nextValid) {
                                    if (nextQVals[na] > maxNextQ) {
                                        maxNextQ = nextQVals[na];
                                    }
                                }
                                if (maxNextQ === -Infinity) maxNextQ = 0.0;
                            }

                            qVals[action] = reward + agent.gamma * maxNextQ;
                        }
                    }
                }
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Simulation / Application State
    // ────────────────────────────────────────────────────────────────────────
    let globalBaseAgent = null;
    let simulations = {};
    
    function smoothData(data, windowSize) {
        let smoothed = [];
        for (let i = 0; i < data.length; i++) {
            let start = Math.max(0, i - windowSize + 1);
            let end = i + 1;
            let window = data.slice(start, end);
            let sum = window.reduce((a, b) => a + b, 0);
            smoothed.push(sum / window.length);
        }
        return smoothed;
    }

    function generateMockData(type, length) {
        let decay = type.split('_')[1];
        const OPTIMAL = 155;
        const DETOUR = 85;
        const CRASH = -20;
        
        let arr = [];
        for(let i=0; i<length; i++) {
            let noise = (Math.random() * 6 - 3);
            
            if (i < 12) {
                arr.push(OPTIMAL + noise);
            } else if (i < 30) {
                if (decay === 'none') {
                    if (i < 15) arr.push(CRASH + noise);
                    else arr.push(DETOUR + noise);
                } else {
                    arr.push(DETOUR + noise);
                }
            } else {
                if (decay === 'none') {
                    arr.push(DETOUR + noise);
                } else if (decay === 'exponential') {
                    if (i < 35) arr.push(DETOUR + (i - 30) * 15 + noise);
                    else arr.push(OPTIMAL + noise);
                } else if (decay === 'linear') {
                    if (i < 45) arr.push(DETOUR + (i - 30) * 4.6 + noise);
                    else arr.push(OPTIMAL + noise);
                } else if (decay === 'cosine') {
                    if (i < 65) arr.push(DETOUR + (i - 30) * 2 + noise);
                    else arr.push(OPTIMAL + noise);
                }
            }
        }
        return smoothData(arr, 5);
    }
    
    let metricsData = {
        none: generateMockData('metrics_none', 100),
        linear: generateMockData('metrics_linear', 100),
        exponential: generateMockData('metrics_exponential', 100),
        cosine: generateMockData('metrics_cosine', 100)
    };
    
    let longTermMetricsData = {
        none: generateMockData('long_none', 100),
        linear: generateMockData('long_linear', 100),
        exponential: generateMockData('long_exponential', 100),
        cosine: generateMockData('long_cosine', 100)
    };
    
    let currentSimQueue = [];
    let isPlaying = false;
    let simIntervalId = null;

    // Interactive Policy Sandbox Grid Editor state
    let sandboxObstacles = [[1, 5], [3, 5]];
    let sandboxWalls = [];

    // Initialize layout grids on load
    const visualEnv = new MazeEnv();
    renderMazeGrid(visualEnv, visualEnv.getState());
    renderPolicyGrid();

    // ────────────────────────────────────────────────────────────────────────
    // UI Event Handlers
    // ────────────────────────────────────────────────────────────────────────
    trainBtn.addEventListener('click', () => {
        trainBtn.disabled = true;
        trainBtn.textContent = "⚡ Training... 0%";
        log("Started training a perfect base agent inside a Web Worker...", "system");
        
        const startTrain = performance.now();
        const worker = new Worker('/static/worker.js');
        
        worker.postMessage({
            lr: parseFloat(lrSlider.value),
            gamma: parseFloat(gammaSlider.value),
            totalEpisodes: parseInt(episodesSlider.value)
        });

        worker.onmessage = (e) => {
            const { type, data } = e.data;
            if (type === 'progress') {
                trainBtn.textContent = `⚡ Training... ${data.percent}%`;
            } else if (type === 'complete') {
                const { baseAgentQTable, metrics, simulationsList } = data;
                
                // Set the global base agent Q-table policy
                globalBaseAgent = new QLearningAgent([0, 1, 2, 3], 'none');
                globalBaseAgent.qTable = baseAgentQTable;
                
                // Store results
                simulations = simulationsList;
                metricsData = metrics.metricsData;
                longTermMetricsData = metrics.longTermMetricsData;
                
                // Re-render charts
                renderChart(metricsData, 'metricsChart');
                renderChart(longTermMetricsData, 'longTermChart');
                
                const endTrain = performance.now();
                log(`Base agent training and multi-agent evaluations completed in ${(endTrain - startTrain).toFixed(2)}ms.`, "success");
                log("Click 'Run Simulation' to watch agents play in real-time!", "success");
                
                prepareSimulationQueue();
                playBtn.disabled = false;
                trainBtn.textContent = "⚡ Train Agent";
                trainBtn.disabled = false;
                
                worker.terminate();
            }
        };
    });

    resetBtn.addEventListener('click', () => {
        log("Resetting sandbox states and clearing obstacles...", "system");
        sandboxObstacles = [[1, 5], [3, 5]];
        sandboxWalls = [];
        globalBaseAgent = null;
        simulations = {};
        currentSimQueue = [];
        isPlaying = false;
        if (simIntervalId) clearInterval(simIntervalId);
        playBtn.disabled = true;
        pauseBtn.disabled = true;
        
        renderMazeGrid(visualEnv, visualEnv.getState());
        renderPolicyGrid();
        
        // Reset metrics UI
        metricDecay.textContent = "-";
        metricIteration.textContent = "-";
        metricStep.textContent = "-";
        metricReward.textContent = "-";
        simTitle.textContent = "Live Simulation: Static Maze";
    });

    shutdownBtn.addEventListener('click', async () => {
        if (confirm("Are you sure you want to shut down the backend server?")) {
            await fetch('/api/shutdown', { method: 'POST' });
            document.body.innerHTML = `
                <div style="text-align: center; margin-top: 20vh; font-family: var(--font-heading); color: var(--text-main);">
                    <h1 style="font-size: 3rem; background: linear-gradient(135deg, #f43f5e, #fb7185); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Server Offline</h1>
                    <p style="margin-top: 15px; color: var(--text-muted)">The backend application has shut down successfully. You can close this page.</p>
                </div>
            `;
        }
    });

    // ────────────────────────────────────────────────────────────────────────
    // Simulation Queue & Loop
    // ────────────────────────────────────────────────────────────────────────
    function prepareSimulationQueue() {
        currentSimQueue = [];
        const decays = [
            { key: 'none', label: 'No Decay' },
            { key: 'linear', label: 'Linear Decay' },
            { key: 'exponential', label: 'Exponential Decay' },
            { key: 'cosine', label: 'Cosine Decay' }
        ];

        for (let dec of decays) {
            let simRuns = simulations[dec.key];
            if (!simRuns) continue;
            for (let i = 0; i < simRuns.length; i++) {
                currentSimQueue.push({
                    decayKey: dec.key,
                    decayLabel: dec.label,
                    iteration: i + 1,
                    simData: simRuns[i]
                });
            }
        }
    }

    playBtn.addEventListener('click', () => {
        if (currentSimQueue.length === 0) {
            prepareSimulationQueue();
        }
        isPlaying = true;
        playBtn.disabled = true;
        pauseBtn.disabled = false;
        runNextSimulation();
    });

    pauseBtn.addEventListener('click', () => {
        isPlaying = false;
        playBtn.disabled = false;
        pauseBtn.disabled = true;
        if (simIntervalId) clearInterval(simIntervalId);
        log("Simulation paused.", "warning");
    });

    function runNextSimulation() {
        if (!isPlaying || currentSimQueue.length === 0) {
            if (currentSimQueue.length === 0) {
                simTitle.textContent = "All Simulations Completed!";
                log("All simulation evaluation queues completed.", "success");
                playBtn.disabled = false;
                pauseBtn.disabled = true;
            }
            return;
        }

        const simItem = currentSimQueue.shift();
        simTitle.textContent = `Simulation: ${simItem.decayLabel} Agent (Iteration ${simItem.iteration}/3)`;
        
        metricDecay.textContent = simItem.decayLabel;
        metricIteration.textContent = `${simItem.iteration}/3`;
        
        log(`Running ${simItem.decayLabel} evaluation loop iteration ${simItem.iteration}/3...`, "system");

        let step = 0;
        const path = simItem.simData.path;
        const rewards = simItem.simData.rewards;
        const dynamicObstacles = simItem.simData.dynamicObstacles;
        
        // Custom playback speed: First 2 slow, next 8 fast
        let baseDelay = parseInt(speedSlider.value);
        const stepDelay = (simItem.iteration <= 2) ? baseDelay * 2 : Math.max(25, baseDelay / 2);
        
        let cumulativeReward = 0;

        simIntervalId = setInterval(() => {
            if (!isPlaying) {
                clearInterval(simIntervalId);
                return;
            }

            const currentPos = path[step];
            const currentDyn = dynamicObstacles[Math.min(step, dynamicObstacles.length - 1)];
            const stepReward = rewards[step] || 0;
            cumulativeReward += stepReward;

            // Update UI Counters
            metricStep.textContent = step;
            metricReward.textContent = cumulativeReward.toFixed(1);

            // Render visual grid
            renderMazeGrid(visualEnv, currentPos, currentDyn);

            // Handle collision indicator
            const nextIdx = step + 1;
            const isLastStep = nextIdx >= path.length;
            
            if (isLastStep && simItem.simData.wasAborted) {
                // Flash red and print crash to log
                collisionFlash.classList.add('active');
                setTimeout(() => collisionFlash.classList.remove('active'), 400);
                log(`Collision! Agent hit dynamic obstacle at [${currentPos[0]}, ${currentPos[1]}]. Instantly replanning...`, "danger");
            }

            step++;
            if (step >= path.length) {
                clearInterval(simIntervalId);
                
                // End step pause before launching next iteration in the queue
                setTimeout(() => {
                    runNextSimulation();
                }, 800);
            }
        }, stepDelay);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Grid Rendering Engines
    // ────────────────────────────────────────────────────────────────────────
    function renderMazeGrid(env, agentPos, activeDynObstacles = []) {
        mazeContainer.innerHTML = '';
        mazeContainer.style.gridTemplateColumns = `repeat(${env.cols}, 36px)`;

        for (let r = 0; r < env.rows; r++) {
            for (let c = 0; c < env.cols; c++) {
                const cell = document.createElement('div');
                cell.className = 'maze-cell';
                let content = '';

                if (env.staticMaze[r][c] === '0') {
                    cell.classList.add('wall');
                } else if (r === env.startPos[0] && c === env.startPos[1]) {
                    cell.classList.add('start');
                } else if (r === env.targetPos[0] && c === env.targetPos[1]) {
                    cell.classList.add('target');
                    content = '🎯';
                }

                // Check dynamic obstacles
                const isObstacle = activeDynObstacles.some(o => o[0] === r && o[1] === c);
                if (isObstacle) {
                    cell.classList.add('dynamic-obs');
                }

                // Check agent position
                if (agentPos && agentPos[0] === r && agentPos[1] === c) {
                    cell.classList.add('agent');
                    const avatar = document.createElement('div');
                    avatar.className = 'agent-avatar';
                    avatar.textContent = '🤖';
                    cell.appendChild(avatar);
                } else if (content) {
                    cell.textContent = content;
                }

                mazeContainer.appendChild(cell);
            }
        }
    }

    function renderPolicyGrid() {
        policyMazeContainer.innerHTML = '';
        policyMazeContainer.style.gridTemplateColumns = `repeat(11, 36px)`;

        // Setup temporary environment matching current sandbox configurations
        const env = new MazeEnv();
        // Override static layout with custom sandbox walls
        for (let w of sandboxWalls) {
            env.staticMaze[w[0]][w[1]] = '0';
        }
        env.buildDistanceMap();

        // Calculate custom policy if base agent exists, else construct dummy agent
        let agent = globalBaseAgent;
        if (!agent) {
            agent = new QLearningAgent([0,1,2,3], 'none');
            agent.epsilon = 0.0;
        }

        // Run Value Iteration to immediately update policy visualization based on obstacles
        replanQTable(agent, env, sandboxObstacles);

        for (let r = 0; r < 11; r++) {
            for (let c = 0; c < 11; c++) {
                const cell = document.createElement('div');
                cell.className = 'maze-cell';
                cell.setAttribute('data-row', r);
                cell.setAttribute('data-col', c);
                let content = '';

                // Render cell type
                if (env.staticMaze[r][c] === '0') {
                    cell.classList.add('wall');
                } else if (r === env.startPos[0] && c === env.startPos[1]) {
                    cell.classList.add('start');
                } else if (r === env.targetPos[0] && c === env.targetPos[1]) {
                    cell.classList.add('target');
                    content = '🎯';
                }

                // Check dynamic obstacles
                const isObstacle = sandboxObstacles.some(o => o[0] === r && o[1] === c);
                if (isObstacle) {
                    cell.classList.add('dynamic-obs');
                }

                // Policy arrow overlay on open paths (excluding start and target)
                const isPath = env.staticMaze[r][c] === '1';
                const isNotEndpoints = !arrayEquals([r,c], env.startPos) && !arrayEquals([r,c], env.targetPos);
                
                if (isPath && isNotEndpoints && !isObstacle) {
                    let qVals = agent.getQValues([r, c]);
                    let valid = env.getValidActions([r, c]);
                    let maxQ = -Infinity;
                    let bestAction = -1;

                    for (let a of valid) {
                        if (qVals[a] > maxQ) {
                            maxQ = qVals[a];
                            bestAction = a;
                        }
                    }

                    if (bestAction !== -1 && maxQ > -500.0) {
                        const arrowSpan = document.createElement('span');
                        arrowSpan.className = 'policy-arrow';
                        const arrows = ['▲', '▶', '▼', '◀'];
                        arrowSpan.textContent = arrows[bestAction];
                        cell.appendChild(arrowSpan);
                    }
                }

                if (content && cell.childNodes.length === 0) {
                    cell.textContent = content;
                }

                // Grid click handler to toggle obstacles/walls
                cell.addEventListener('click', () => {
                    const row = parseInt(cell.getAttribute('data-row'));
                    const col = parseInt(cell.getAttribute('data-col'));

                    // Protect start and target positions
                    if (arrayEquals([row, col], env.startPos) || arrayEquals([row, col], env.targetPos)) {
                        return;
                    }

                    const mode = document.querySelector('input[name="clickMode"]:checked').value;

                    if (mode === 'dynamic') {
                        // Toggle dynamic obstacle
                        const idx = sandboxObstacles.findIndex(o => o[0] === row && o[1] === col);
                        if (idx !== -1) {
                            sandboxObstacles.splice(idx, 1);
                        } else {
                            sandboxObstacles.push([row, col]);
                            // Remove static wall if present
                            const wIdx = sandboxWalls.findIndex(w => w[0] === row && w[1] === col);
                            if (wIdx !== -1) sandboxWalls.splice(wIdx, 1);
                        }
                    } else if (mode === 'wall') {
                        // Toggle static wall
                        const idx = sandboxWalls.findIndex(w => w[0] === row && w[1] === col);
                        if (idx !== -1) {
                            sandboxWalls.splice(idx, 1);
                        } else {
                            sandboxWalls.push([row, col]);
                            // Remove dynamic obstacle if present
                            const oIdx = sandboxObstacles.findIndex(o => o[0] === row && o[1] === col);
                            if (oIdx !== -1) sandboxObstacles.splice(oIdx, 1);
                        }
                    }

                    // Re-render policy view reflecting changes
                    renderPolicyGrid();
                });

                policyMazeContainer.appendChild(cell);
            }
        }
    }



    // ────────────────────────────────────────────────────────────────────────
    // Chart Rendering
    // ────────────────────────────────────────────────────────────────────────
    let chartInstances = {};

    function renderChart(metrics, canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        
        if (chartInstances[canvasId]) {
            chartInstances[canvasId].destroy();
        }

        const stepVal = (canvasId === 'longTermChart') ? 50 : 1;
        const labels = Array.from({ length: metrics.none.length }, (_, i) => (i + 1) * stepVal);
        
        const datasets = [
            {
                label: 'No Decay',
                data: metrics.none,
                borderColor: '#f43f5e',
                backgroundColor: 'rgba(244, 63, 94, 0.05)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 1
            },
            {
                label: 'Linear Decay',
                data: metrics.linear,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.05)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 1
            },
            {
                label: 'Exponential Decay',
                data: metrics.exponential,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.05)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 1
            },
            {
                label: 'Cosine Decay',
                data: metrics.cosine,
                borderColor: '#eab308',
                backgroundColor: 'rgba(234, 179, 8, 0.05)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 1
            }
        ];

        chartInstances[canvasId] = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#cbd5e1',
                            font: { family: 'Plus Jakarta Sans', weight: '600', size: 12 }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#64748b', font: { family: 'Plus Jakarta Sans' } },
                        grid: { color: 'rgba(255,255,255,0.03)' }
                    },
                    y: {
                        ticks: { color: '#64748b', font: { family: 'Plus Jakarta Sans' } },
                        grid: { color: 'rgba(255,255,255,0.03)' }
                    }
                }
            }
        });
    }

    // Instantly render pre-fed rich metrics on load
    renderChart(metricsData, 'metricsChart');
    renderChart(longTermMetricsData, 'longTermChart');
});
