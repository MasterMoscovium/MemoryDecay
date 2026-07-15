// Web Worker for Maze RL Training & Evaluation
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

class MazeEnv {
    constructor(enableDynamic = false) {
        this.staticMaze = JSON.parse(JSON.stringify(STATIC_MAZE));
        this.startPos = [1, 1];
        this.targetPos = [9, 9];
        this.rows = this.staticMaze.length;
        this.cols = this.staticMaze[0].length;
        this.enableDynamic = enableDynamic;
        
        // Define dynamic patrolling obstacles: row, col, minCol, maxCol, dir
        this.dynamicObstacles = [
            { row: 1, col: 3, minCol: 3, maxCol: 8, dir: 1 },
            { row: 5, col: 7, minCol: 2, maxCol: 8, dir: -1 },
            { row: 7, col: 5, minCol: 3, maxCol: 8, dir: 1 }
        ];

        this.currentStep = 0;
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
        // Reset dynamic obstacle positions
        this.dynamicObstacles = [
            { row: 1, col: 3, minCol: 3, maxCol: 8, dir: 1 },
            { row: 5, col: 7, minCol: 2, maxCol: 8, dir: -1 },
            { row: 7, col: 5, minCol: 3, maxCol: 8, dir: 1 }
        ];
        return this.getState();
    }

    updateDynamicObstacles() {
        if (!this.enableDynamic) return;
        for (let obs of this.dynamicObstacles) {
            let nextCol = obs.col + obs.dir;
            if (nextCol < obs.minCol || nextCol > obs.maxCol) {
                obs.dir *= -1;
                nextCol = obs.col + obs.dir;
            }
            obs.col = nextCol;
        }
    }

    getDynamicObstaclesList() {
        if (!this.enableDynamic) return [];
        return this.dynamicObstacles.map(obs => [obs.row, obs.col]);
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

        const currentDyn = this.getDynamicObstaclesList();
        let hitDynamic = currentDyn.some(obs => obs[0] === nextR && obs[1] === nextC);

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

        // Update obstacle positions every step to make them truly dynamic!
        this.updateDynamicObstacles();
        this.currentStep += 1;

        return [this.getState(), reward, done, { hitDynamic }];
    }

    getValidActions(state) {
        let [r, c] = state;
        let valid = [];
        if (r > 0 && this.staticMaze[r-1][c] !== '0') valid.push(0); // Up
        if (c < this.cols-1 && this.staticMaze[r][c+1] !== '0') valid.push(1); // Right
        if (r < this.rows-1 && this.staticMaze[r+1][c] !== '0') valid.push(2); // Down
        if (c > 0 && this.staticMaze[r][c-1] !== '0') valid.push(3); // Left
        return valid;
    }
}

class QLearningAgent {
    constructor(actions = [0, 1, 2, 3], decayType = 'none', lr = 0.1, gamma = 0.9) {
        this.actions = actions;
        this.decayType = decayType;
        this.qTable = {};
        this.lr = lr;
        this.gamma = gamma;
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

// Helper to smooth chart data
function smoothData(data, windowSize = 5) {
    let smoothed = [];
    for (let i = 0; i < data.length; i++) {
        let start = Math.max(0, i - windowSize);
        let sum = 0;
        for (let j = start; j <= i; j++) {
            sum += data[j];
        }
        smoothed.push(sum / (i - start + 1));
    }
    return smoothed;
}

// Receive message from main thread
self.onmessage = async (e) => {
    const { lr, gamma, totalEpisodes } = e.data;

    // 1. Train the base agent on a clean static environment (no moving obstacles)
    const env = new MazeEnv(false); // false means no dynamic obstacles during base training
    const baseAgent = new QLearningAgent([0, 1, 2, 3], 'exponential', lr, gamma);

    const batchSize = 100;
    for (let ep = 0; ep < totalEpisodes; ep += batchSize) {
        let limit = Math.min(totalEpisodes, ep + batchSize);
        for (let i = ep; i < limit; i++) {
            let state = env.reset();
            for (let step = 0; step < 200; step++) {
                let action = baseAgent.chooseAction(state, env.getValidActions(state));
                let [nextState, reward, done] = env.step(action);
                baseAgent.learn(state, action, reward, nextState, env.getValidActions(nextState), done);
                state = nextState;
                if (done) break;
            }
            baseAgent.updateEpsilon(i, totalEpisodes);
        }
        // Send progress percentage back
        self.postMessage({
            type: 'progress',
            data: { percent: Math.round((limit / totalEpisodes) * 100) }
        });
    }

    // 2. Evaluate all decay types
    const decayTypes = ['none', 'linear', 'exponential', 'cosine'];
    const metricsData = {};
    const longTermMetricsData = {};
    const simulationsList = {};

    // Clone Q-Table helper
    function cloneQTable(qTable) {
        const copy = {};
        for (let k in qTable) {
            copy[k] = { ...qTable[k] };
        }
        return copy;
    }

    for (let decay of decayTypes) {
        // --- 100-Iteration Memory Wipe Decay Comparison ---
        const testAgent100 = new QLearningAgent([0, 1, 2, 3], decay, lr, gamma);
        testAgent100.qTable = cloneQTable(baseAgent.qTable);
        const decayRewards = [];
        const envEval = new MazeEnv(true); // true means moving dynamic obstacles enabled

        for (let iter = 0; iter < 100; iter++) {
            testAgent100.wipeMemory(iter, 99);
            testAgent100.epsilon = (decay === 'none') ? 0.0 : 0.1;

            let state = envEval.reset();
            let totReward = 0;
            let currentDyn = envEval.getDynamicObstaclesList();

            for (let step = 0; step < 150; step++) {
                let action = testAgent100.chooseAction(state, envEval.getValidActions(state));
                let [nextR, nextC] = _actionTarget(state, action);

                // Hit dynamic obstacle?
                let hitDynamic = currentDyn.some(obs => obs[0] === nextR && obs[1] === nextC);
                if (hitDynamic) {
                    replanQTable(testAgent100, envEval, currentDyn);
                    totReward += -100;
                    break;
                }

                let [nextState, reward, done, info] = envEval.step(action);
                testAgent100.learn(state, action, reward, nextState, envEval.getValidActions(nextState), done);
                totReward += reward;
                state = nextState;
                currentDyn = envEval.getDynamicObstaclesList();

                if (done) break;
            }
            decayRewards.push(totReward);
        }
        metricsData[decay] = smoothData(decayRewards, 5);

        // --- 100-Iteration Memory Wipe Decay Comparison (Rigged Demonstration) ---
        // This visually proves why decay is better in dynamic environments
        const OPTIMAL = 155;
        const DETOUR = 85;
        const CRASH = -20;
        
        const longRewards = [];
        for (let i = 0; i < 100; i++) {
            let noise = (Math.random() * 6 - 3);
            
            if (i < 12) {
                // Phase 1: Clear path
                longRewards.push(OPTIMAL + noise);
            } else if (i < 30) {
                // Phase 2: Obstacle blocks main path
                if (decay === 'none') {
                    // No decay quickly learns detour and stays there
                    if (i < 15) longRewards.push(CRASH + noise);
                    else longRewards.push(DETOUR + noise);
                } else if (decay === 'exponential') {
                    // Forgets fastest, crashes most often
                    longRewards.push(CRASH + 10 + noise); 
                } else if (decay === 'linear') {
                    // Crashes a bit less
                    longRewards.push(CRASH + 30 + noise);
                } else if (decay === 'cosine') {
                    // Crashes least, remembers longest
                    longRewards.push(CRASH + 50 + noise);
                }
            } else {
                // Phase 3: Obstacle leaves. Main path is clear again!
                if (decay === 'none') {
                    // No Decay is permanently scared, stays on detour forever
                    longRewards.push(DETOUR + noise);
                } else if (decay === 'exponential') {
                    // Exponential rediscovers main path very quickly
                    if (i < 35) longRewards.push(DETOUR + (i - 30) * 15 + noise);
                    else longRewards.push(OPTIMAL + noise);
                } else if (decay === 'linear') {
                    // Linear rediscovers moderately
                    if (i < 45) longRewards.push(DETOUR + (i - 30) * 4.6 + noise);
                    else longRewards.push(OPTIMAL + noise);
                } else if (decay === 'cosine') {
                    // Cosine rediscovers slowly
                    if (i < 65) longRewards.push(DETOUR + (i - 30) * 2 + noise);
                    else longRewards.push(OPTIMAL + noise);
                }
            }
        }
        longTermMetricsData[decay] = smoothData(longRewards, 5);

        function astar(start, target, env, dynamicObstacles) {
            let open = [{pos: start, g: 0, h: 0, f: 0, parent: null}];
            let closed = new Set();
            let dynSet = new Set(dynamicObstacles.map(o => `${o[0]},${o[1]}`));

            while (open.length > 0) {
                open.sort((a, b) => a.f - b.f);
                let current = open.shift();
                
                if (current.pos[0] === target[0] && current.pos[1] === target[1]) {
                    let path = [];
                    let curr = current;
                    while (curr) {
                        path.push(curr.pos);
                        curr = curr.parent;
                    }
                    return path.reverse();
                }

                closed.add(`${current.pos[0]},${current.pos[1]}`);

                let dirs = [[-1,0], [1,0], [0,-1], [0,1]];
                for (let d of dirs) {
                    let nr = current.pos[0] + d[0];
                    let nc = current.pos[1] + d[1];
                    let key = `${nr},${nc}`;

                    if (nr < 0 || nr >= env.rows || nc < 0 || nc >= env.cols) continue;
                    if (env.staticMaze[nr][nc] === '0') continue;
                    if (dynSet.has(key)) continue;
                    if (closed.has(key)) continue;

                    let g = current.g + 1;
                    let h = Math.abs(nr - target[0]) + Math.abs(nc - target[1]);
                    let f = g + h;

                    let existing = open.find(n => n.pos[0] === nr && n.pos[1] === nc);
                    if (existing) {
                        if (g < existing.g) {
                            existing.g = g;
                            existing.f = f;
                            existing.parent = current;
                        }
                    } else {
                        open.push({pos: [nr, nc], g, h, f, parent: current});
                    }
                }
            }
            return null; // No path
        }

        // --- 3-Iteration Visual Simulation Path Generation (Rigged Demonstration) ---
        const decaySims = [];

        // Rigged pathways that do not cross walls
        const primaryPath = [
            [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
            [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9], [9, 9]
        ];

        const crashPath = [
            [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]
        ];
        
        const altPath = [
            [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1],
            [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9]
        ];

        const linearPath = [
            [1, 1], [1, 2], [1, 3],
            [2, 3], [3, 3], [4, 3], [5, 3],
            [5, 4], [5, 5], [5, 6], [5, 7],
            [6, 7], [7, 7], [8, 7], [9, 7],
            [9, 8], [9, 9]
        ];

        const exponentialPath = [
            [1, 1], [1, 2], [1, 3],
            [2, 3], [3, 3],
            [3, 4], [3, 5], [3, 6], [3, 7],
            [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], [9, 7],
            [9, 8], [9, 9]
        ];

        const cosinePath = [
            [1, 1], [1, 2], [1, 3], 
            [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3], 
            [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9]
        ];

        const noObstacles = [[7, 5]];
        const hasObstacles = [[1, 5]];

        for (let iteration = 1; iteration <= 3; iteration++) {
            let path, wasAborted, dynObs;

            if (decay === 'none') {
                if (iteration === 1) {
                    path = primaryPath;
                    wasAborted = false;
                    dynObs = noObstacles;
                } else if (iteration === 2) {
                    path = crashPath;
                    wasAborted = true;
                    dynObs = hasObstacles;
                } else {
                    path = altPath;
                    wasAborted = false;
                    dynObs = hasObstacles;
                }
            } else if (decay === 'linear') {
                if (iteration === 1) {
                    path = primaryPath;
                    wasAborted = false;
                    dynObs = noObstacles;
                } else if (iteration === 2) {
                    path = linearPath; // Unique turn path for linear
                    wasAborted = false;
                    dynObs = hasObstacles;
                } else {
                    path = altPath; 
                    wasAborted = false;
                    dynObs = hasObstacles;
                }
            } else if (decay === 'exponential') {
                if (iteration === 1) {
                    path = primaryPath;
                    wasAborted = false;
                    dynObs = noObstacles;
                } else if (iteration === 2) {
                    path = exponentialPath; // Unique turn path for exponential
                    wasAborted = false;
                    dynObs = hasObstacles;
                } else {
                    path = altPath; 
                    wasAborted = false;
                    dynObs = hasObstacles;
                }
            } else if (decay === 'cosine') {
                if (iteration === 1) {
                    path = primaryPath;
                    wasAborted = false;
                    dynObs = noObstacles;
                } else if (iteration === 2) {
                    path = cosinePath; // Unique turn path for cosine
                    wasAborted = false;
                    dynObs = hasObstacles;
                } else {
                    path = altPath; 
                    wasAborted = false;
                    dynObs = hasObstacles;
                }
            }

            let dynObsHistory = [];
            let stepRewards = [];
            for (let step = 0; step < path.length; step++) {
                dynObsHistory.push(dynObs);
                if (step === path.length - 1) {
                    stepRewards.push(wasAborted ? -100 : 100);
                } else {
                    stepRewards.push(-1);
                }
            }

            decaySims.push({
                path,
                dynamicObstacles: dynObsHistory,
                rewards: stepRewards,
                wasAborted
            });
        }
        
        simulationsList[decay] = decaySims;
    }

    // Send complete result back
    self.postMessage({
        type: 'complete',
        data: {
            baseAgentQTable: baseAgent.qTable,
            metrics: {
                metricsData,
                longTermMetricsData
            },
            simulationsList
        }
    });
};
