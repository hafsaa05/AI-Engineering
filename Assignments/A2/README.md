# Rubik's Cube Race

AI Engineering Assignment 02 | Hafsa Rashid | 23K-0064

## 🎥 Demo Video

**Watch online:** [Google Drive link](https://drive.google.com/file/d/1KWW2x6Ek-NvBgvku_ftuBjQFE8_uBk8m/view?usp=sharing)

**In this repository:** [demo/AIE_A2_Demo_HafsaRashid_23K-0064.mp4](demo/AIE_A2_Demo_HafsaRashid_23K-0064.mp4)

Model IDs verified on the OpenRouter model pages on 21 Oct 2026. Jev, Luna and Clef run on the Decisions API; Opus runs on chat completions.

## What this project does

This project races four AI models against each other on the same scrambled 3x3 Rubik's Cube. Each model sees the cube and picks one move at a time, until the cube is solved or it runs out of moves or time. Every move is timed and logged, so afterwards the models can be compared on:

- **Speed:** average latency per move and total time per run
- **Quality:** whether it solved the cube within 20 moves, how many moves it used, and the optimality gap (moves used minus the shortest possible solution)

The four models are:

| Model | Type | API |
|---|---|---|
| Jev 1.13 | Decision model | OpenRouter Decisions API |
| GPT-6 Luna Decisions | Decision model | OpenRouter Decisions API |
| Cloudflare Clef | Decision model | OpenRouter Decisions API |
| Claude Opus 5.5 | LLM | OpenRouter chat completions |

Everything is on one page: the live race (four panels, one cube per model) at the top, and below it the results with tables, charts and a slide for the final results.

## Teacher's rules for the OpenRouter API key

These rules come from my teacher and are enforced in the backend code, not only in the GUI:

1. **Nobody but me starts a real OpenRouter run.** All tests use mocks or fake responses and are blocked from the network. The real run is started once, by me, from the GUI.
2. **Mode switch.** The **Mode** dropdown on the race screen switches between Ollama and OpenRouter while the server runs. `MODE` in `.env` (`ollama` by default, `mock` for testing) is only the starting value. Having a key in `.env` does not switch on real mode by itself: real calls need OpenRouter mode **and** the key. The mode can not change while a race is running, and it is always shown on the page.
3. **Puzzles.** The `openrouter` set has exactly 1 puzzle, 3 moves from solved. The `ollama` set has 5 puzzles, each 3 moves from solved. The older sets stay for mock testing.
4. **Move limit:** 20 moves per run, in every mode.
5. **Hard request limits (OpenRouter only):** Claude at most 5 requests in total, each decision model at most 20, all together at most 65, for the whole experiment. Failed requests and retries count too. OpenRouter has 0 retries, so an invalid answer just uses up a move. A model that reaches its limit stops at once and is marked NOT SOLVED with the reason **CALL LIMIT REACHED**.
6. **Persistent counter** in `data/api_usage.json`, checked before every request, never reset (there is no reset button).
7. **Cache** in `data/api_cache.json`. A request that was already answered is never sent again and does not count.
8. **Logging** of latency, cache hits, request numbers and the real cost from the API's `usage` field.
9. **Safety.** The real race only starts after a confirmation question, and there is a **Dry run** button to check everything first.

## Setup

You need Python 3.10 or newer.

1. Open a terminal in the `rubiks-cube-race` folder.

2. Create and turn on a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)
   source .venv/bin/activate       (Mac / Linux)
   ```

3. Install the packages:

   ```
   pip install -r requirements.txt
   ```

4. Make your `.env` file from the example:

   ```
   copy .env.example .env          (Windows)
   cp .env.example .env            (Mac / Linux)
   ```

   Add the OpenRouter key if you have one. `MODE` is optional: it is only the starting mode (default `ollama`), and you can switch modes later from the GUI (see [Modes](#modes-mock-ollama-and-openrouter) below). The key is only needed for OpenRouter mode. It is only read by the backend, never goes to the browser, and `.env` is in `.gitignore`.

## How to run

Start the server from the `rubiks-cube-race` folder:

```
uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser. (On Windows, use `127.0.0.1` instead of `localhost`, because `localhost` can add a 2 second delay on every connection.)

To run the tests:

```
python -m pytest
```

There are also two command line tools:

```
python -m backend.puzzles             # regenerate data/puzzles.json (same seed, same puzzles)
python -m backend.benchmark primary   # run a whole set (or one puzzle id) and print a summary
```

## Using the app

### Race screen

1. Choose the **Mode**: Ollama (local) or OpenRouter (real API). The four panels, the Puzzle list, the saved races and the results section switch to that mode right away.
2. Pick a puzzle from the **Puzzle** list (Ollama mode: `ollama-1` to `ollama-5`, OpenRouter mode: only `openrouter-1`). The scramble and its optimal distance are shown underneath.
3. Click **Run Live Race**. All four models start at the same time (in Ollama mode they take turns). Each panel shows the cube, a timer, the move count, the last move and a status (thinking... / deciding... / SOLVED / NOT SOLVED).
4. When everyone has finished, each panel gets a rank medal (1st, 2nd, 3rd, 4th). The winner gets a gold border, and models that did not solve get a red **RAN OUT OF MOVES** or **OUT OF TIME** label.
5. **2D net** on a panel opens a flat view of all six faces of that model's cube.
6. Every race is saved. Pick it from **Saved race**, choose a speed (1x, 2.5x, 5x) and click **Replay** to watch it again. The replay uses the real recorded times, so it is good for the screen recording.

Only one race (or benchmark) can run at a time. If you try to start another, you get a message telling you to wait.

A small label next to the Mode dropdown shows the mode. In OpenRouter mode it turns into a red **Real API mode** label, and the top bar gets a red line, so it can never be missed. Each model panel shows its own API calls, cache hits, cost, average latency per move and optimality gap at the bottom of its stat card.

**The real OpenRouter run** (only in OpenRouter mode, with the key set in `.env`):

1. First click **Dry run (OpenRouter)**. It runs the whole OpenRouter race (same requests, same limits, same counting) but never sends anything. It works in any mode, even without a key. At the end it says how many calls a real run would use. Nothing is saved and the real counter does not change.
2. Then pick the `openrouter-1` puzzle and click **Run Live Race**. A box asks: *"This will use up to X real API calls. Calls used so far: Y / 65. Continue?"* Only **OK** starts the race. In OpenRouter mode the backend only allows this one puzzle, and refuses any race that was not confirmed.
3. Cached moves show a dashed gold border in the move history.

### Results (below the race grid)

Scroll down under the four panels.

- Choose a **Puzzle set** and **Models** (mock, Ollama or OpenRouter results).
- **Run benchmark on this set** runs every puzzle in the set, one race after another, and shows the progress. It is switched off in OpenRouter mode, so the API budget can only be used through the confirmed race.
- The page shows a summary table per model, four bar charts (latency, total time, success rate, optimality gap), automatic **Key findings**, and a per puzzle table with each model's moves and result.
- **Slide mode** shows the summary, charts and findings on one full screen, for the end of the recording. Press Esc or **Exit slide mode** to go back.

## Modes: mock, Ollama and OpenRouter

The **Mode** dropdown on the race screen decides which four models race. The choice is kept in the server's memory (`POST /api/mode`), so no restart is needed. `MODE` in `.env` is only the mode the server starts in (default `ollama`); the command line tool also reads it. The GUI only offers Ollama and OpenRouter: mock mode is for the tests.

- **Mock** (tests only, or `MODE=mock` for the command line): the four lanes are fake models, so everything can be built and tested for free.
  - `mock-random` plays random moves instantly (a fast, bad model)
  - `mock-oracle` plays the BFS optimal move with a 0.3s delay (a slow, perfect model)
  - `mock-noisy` plays the oracle move 70% of the time and a random move otherwise
  - `mock-broken` sometimes returns junk like `"R3"` or `None`, to test retries and logging


- **Ollama**: four local models run through [Ollama](https://ollama.com), with no API key and no OpenRouter calls at all.

  | Model | Ollama tag | Type |
  |---|---|---|
  | Clef Flash | `clef-flash:latest` | Decision model (10 GB, slow to load) |
  | Qwen3.5 0.8B | `qwen3.5:0.8b` | LLM |
  | DeepSeek-R1 1.5B | `deepseek-r1:1.5b` | LLM (thinking model) |
  | Gemma3 1B | `gemma3:1b` | LLM |

  **Installing Ollama and the models:**

  1. Download and install Ollama from [ollama.com/download](https://ollama.com/download), then start the Ollama app (or run `ollama serve`).
  2. Pull the four models (about 13 GB in total):

     ```
     ollama pull clef-flash:latest
     ollama pull qwen3.5:0.8b
     ollama pull deepseek-r1:1.5b
     ollama pull gemma3:1b
     ```

  3. Check with `ollama list`. If Ollama is not running, or a model is missing, the race screen shows a clear message (for example "Run: ollama pull gemma3:1b") instead of crashing.

  How Ollama mode works:
  - **Same questions as OpenRouter:** the three LLMs get exactly the same messages as Claude, with the same JSON format. Clef Flash is a decision model: `ollama show clef-flash` lists its capabilities as `decision` and `vision`, so it cannot chat. It is asked through Ollama's decision endpoint (`/v1/systemone`, the one `ollama run --ask --choice` uses), which takes exactly the same choice question as the OpenRouter decision models.
  - **One at a time:** a laptop can only hold one model in memory, so the models take turns. Each model is loaded before its turn and unloaded after it, and loading is not counted in the timings.
  - **Timeout:** 120s per move, and it is a hard limit. The 120s is for the whole move, retries included, so a model that keeps timing out can not stretch one move to 6 minutes. An answer that only arrives after the limit (for example because the computer was paused) is not accepted. While a race runs, the app also keeps Windows from going to sleep.
  - **Run time:** like every mode, each model gets exactly 5 minutes for its run. A move still running at 5:00 is cancelled and the run ends as OUT OF TIME.
  - **Turn order:** Gemma3 1B, Qwen3.5 0.8B, DeepSeek-R1 1.5B, then Clef Flash. The panels keep their usual positions.
  - **Output cap:** every model gets the same cap of 4096 output tokens, so DeepSeek-R1 has room to finish thinking and still answer.
  - **Thinking on or off:** Qwen3.5 and Gemma3 run with `think: false`. They are our small, fast models, and in a test race Qwen3.5 spent its whole 120s thinking without ever answering. DeepSeek-R1 keeps thinking on (`think: true`) on purpose: it is our "slow thinker", so the race shows whether thinking first helps.
  - **Forced JSON:** the three LLMs get Ollama's `format` parameter with a JSON schema that only allows `{"move": <one of the 18 moves>}`. Clef Flash answers a choice question, so its answer is always one of the 18 moves too.
  - **Thinking models:** the reasoning comes either in a separate `thinking` field or as a `<think>...</think>` block at the start of the reply. Only the final answer after it is read, and it is checked as strictly as every other model's.
  - **No API limits** apply to Ollama, but the same 20 move limit does.
  - **Replays:** a replay of an Ollama race shows the four models side by side, using their recorded times.

- **OpenRouter**: the four real API models are used. The dropdown only switches to it when `OPENROUTER_API_KEY` is in `.env` (restart the server after adding the key). Without the key the OpenRouter adapters refuse to start. The command line tool also refuses this mode: the real run is only started from the GUI.

Results from each mode are kept apart: the results section always shows the current mode's puzzle set and models, and mock result files are ignored by git.

Note: the OpenRouter model IDs for GPT-6 Luna Decisions and Cloudflare Clef could not be checked before getting a key. If a model gives an error, check its ID in `backend/config.py`.

## How it works

**Same puzzles for everyone.** The puzzles are made once from a fixed random seed and saved in `data/puzzles.json`: 3 warm up (3 to 4 turns), 10 primary (4 to 5 turns), 3 extension (15 to 20 turns), 1 openrouter and 5 ollama (each exactly 3 moves from solved, checked with BFS). A scramble never turns the same face twice in a row. In a race, every model gets the same starting cube, the same question wording, the same last 5 moves as history, and the same timeout and retry rules.

**The cube.** The cube state is a 54 letter string (9 stickers for each face, in the order U R F D L B). The `magiccube` library is used to learn what each of the 18 moves does to the 54 stickers. After that, a move is just reordering the string, which is very fast. The tests check every move against magiccube.

**BFS for the shortest solution.** For warm up and primary puzzles, a bidirectional BFS finds the exact shortest solution. It searches forward from the scrambled cube and backward from the solved cube at the same time, and stops when the two searches meet. It skips moves on the same face as the previous move. Each side only needs to go about half the depth, so even depth 6 takes a fraction of a second. The shortest distance is stored with the puzzle and used for the optimality gap. Extension puzzles are too deep for BFS, so they have no optimal distance.

**Asking the models.**
- The decision models get a "choice" question whose 18 options are the moves, each with a plain description (for example "Turn the Right face a quarter turn (90 degrees) clockwise").
- Opus gets the same text through chat completions and must reply with exactly `{"move": "..."}`. A JSON schema (structured output) makes the API enforce this format.

**Invalid moves.** Every answer is checked against the 18 moves exactly. Nothing is fixed for the model: `"r"`, `" R"` or `"R3"` are all invalid.
- If the answer is invalid, or the call fails (timeout, network error), the model is asked again: up to 2 retries for mock and Ollama, 0 retries for OpenRouter.
- If it is still invalid after the retries, the move counts as one of the 20 moves, but the cube does not change.
- Every attempt is logged with its latency, output and error.

**Limits.** Each call has a 60 second timeout (Ollama: 120 seconds for the whole move, retries included), each run has at most 20 moves, and each run has a hard 5 minute total limit (a move still running at 5:00 is cancelled, and the clock never shows more than 5:00.00). A run ends as **SOLVED**, **RAN OUT OF MOVES**, **OUT OF TIME** or (OpenRouter only) **CALL LIMIT REACHED**. Every loop in the code has a fixed maximum, so nothing can run forever.

**How the API limits and the cache work.** Every OpenRouter request goes through one function in `backend/api_guard.py`, always in this order:

1. **Cache:** the request body is turned into a key (model + a SHA-256 hash of the exact body). If `data/api_cache.json` already has an answer for that key, it is reused: no call, nothing counted, logged as a cache hit.
2. **Limits:** `data/api_usage.json` is read from disk (so a restart changes nothing). If the model has used its requests (Claude 5, each decision model 20) or all models together have used 65, the request is refused and the model stops with CALL LIMIT REACHED.
3. **Count:** one request is added and saved to disk **before** it is sent, so even a request that crashes or times out is counted.
4. **Send:** the HTTP call is made, the real cost is read from the response's `usage` field, and the response is saved in the cache.

If the counter file is ever unreadable, every request is refused (it never falls back to "nothing used"). A dry run does steps 1 to 3 on a copy of the counts that is never saved and makes up an answer instead of step 4, so it can never send anything or change the real counter or cache. With 20 moves and 0 retries a model makes at most 20 requests per race, so the largest possible first race is 20 + 20 + 20 + 5 = 65.

**Ranking.** Every model gets a rank from 1 to 4:
1. Solved models come first, ordered by fewer moves, then less time.
2. Unsolved models come next, ordered by how many stickers are in the correct place (more is better), then less time.

**Saved results.** Each run is saved as a JSON file in `results/` with the model, puzzle, scramble, every move, per move latency, retries, invalid outputs, total time, solved, end reason, optimal distance, optimality gap and confidence (the decision models return a confidence value). For OpenRouter every attempt also has: live API latency or cached response time (separate fields), cache hit yes/no, the request number in the whole experiment, and the cost. Each run adds up its API calls, cache hits and total cost.

## Results

Both runs used a puzzle that is exactly 3 moves from solved (BFS optimum = 3). Every model got the same puzzle, the same question and the same 20 move limit, with a hard 5 minute limit per model. "Correct stickers" is how many of the 54 stickers are in the right place at the end (more is better); it is what ranks unsolved models.

**OpenRouter run (openrouter-1, 3 moves from solved, 20 move limit)**

All four models were run on the same puzzle with the same rules. The OpenRouter key had strict limits (65 requests in total, 5 for Claude, and a small credit balance), and this affected two of the models:

- Jev 1.13 played all 20 moves: 9 API calls and 11 cache hits, 3.43s per move on average, 68.52s in total, cost $0.0003.
- Cloudflare Clef played all 20 moves: 14 API calls and 6 cache hits, 3.07s per move on average, 61.42s in total, cost $0.0009.
- GPT-6 Luna Decisions used 20 API calls, but the provider rejected the requests (HTTP 502), so it could not make a valid move.
- Claude Opus 5.5 did not make a valid move: its requests were rejected because of the key’s credit limit (HTTP 402), and it then reached the 5 call Claude limit.

| Rank | Model | Type | End reason | Moves (valid) | Avg latency / move | Total time | Correct stickers | API calls | Cache hits | Failed calls | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Claude Opus 5.5 | Frontier LLM | CALL LIMIT REACHED | 5 (0) | 1.98s | 9.9s | 23 | 5 | 0 | 5 | $0.0000 |
| 2 | Jev 1.13 | Decision model | RAN OUT OF MOVES | 20 (18) | 3.43s | 68.5s | 23 | 9 | 11 | 2 | $0.0003 |
| 3 | GPT-6 Luna Decisions | Decision model | RAN OUT OF MOVES | 20 (0) | 6.65s | 133.0s | 23 | 20 | 0 | 20 | $0.0000 |
| 4 | Cloudflare Clef | Decision model | RAN OUT OF MOVES | 20 (19) | 3.07s | 61.4s | 20 | 14 | 6 | 1 | $0.0009 |

In total the run used 48 of the 65 allowed requests, with 17 cache hits and a total logged cost of about $0.0013. The cache saved 17 paid requests, which shows why caching mattered with such a limited key.

**Ollama results (Ollama set, local models on a laptop CPU)**

7 runs per model: the 5 benchmark puzzles (ollama-1 to ollama-5) plus 2 earlier live races on ollama-1.

| Model | Type | Runs | Success rate | Avg latency / move | Avg total time | Avg moves | Valid moves | Failed calls | Invalid outputs | End reason |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma3 1B | LLM | 7 | 0% | 5.54s | 110.8s | 20.0 | 140 | 0 | 0 | RAN OUT OF MOVES (7 of 7) |
| Qwen3.5 0.8B | LLM | 7 | 0% | 7.32s | 146.4s | 20.0 | 140 | 0 | 0 | RAN OUT OF MOVES (7 of 7) |
| DeepSeek-R1 1.5B | LLM (thinking) | 7 | 0% | 57.84s | 300.0s | 4.6 | 29 | 4 | 0 | OUT OF TIME (7 of 7) |
| Clef Flash | Decision model | 7 | 0% | 95.90s | 300.0s | 2.4 | 0 | 48 | 0 | OUT OF TIME (7 of 7) |

Ollama has no API calls or cost, and every puzzle was 3 moves from solved, so no optimality gap could be calculated. Clef Flash never answered in any run: all 48 of its calls failed because Ollama's model server crashed, ran out of memory or timed out, so its 95.9s per move is time spent on failed calls. DeepSeek-R1 thinks before every answer, so it only managed 3 to 6 moves in 5 minutes.

Key findings:

- Fastest model: on OpenRouter, Cloudflare Clef was the fastest model that actually answered (3.07s per move). On Ollama, Gemma3 1B (5.54s per move).
- Highest solve rate: no model solved the puzzle (0% for all eight models).
- Closest to optimal: not defined, because no model solved the puzzle.
- Was the fastest also the best? Can't tell, because no model solved the puzzle. No model ended with more correct stickers than it started with.

Warm up and extension notes: the real models only ran on `openrouter-1` (OpenRouter) and `ollama-1` to `ollama-5` (Ollama), because of the API call limits and the slow local models. The warm up, primary and extension sets were used with the mock models for testing.

## Answers to the 5 questions

**1. Which model is fastest in terms of making decisions?**

On OpenRouter, Cloudflare Clef was the fastest model that actually answered, at about 3.07s per move, followed by Jev 1.13 at 3.43s. On Ollama (laptop CPU), Gemma3 1B was the fastest at 5.54s per move, then Qwen3.5 0.8B at 7.32s. DeepSeek-R1 1.5B (57.8s) and Clef Flash (95.9s) were much slower. DeepSeek is slow because it thinks before answering. Clef Flash is a large model for a laptop, and it never actually answered: all of its calls failed, so its 95.9s is time spent on failed calls, not on decisions.

**2. Which model achieved the highest solve rate?**

No model solved the cube, so the solve rate was 0% for every model, even though each cube was only 3 moves away from solved. Jev, Clef, Gemma and Qwen used all 20 moves, DeepSeek-R1 and Clef Flash ran out of time, and on OpenRouter, Luna and Opus could not make a valid move because of API limits.

**3. Which model found the shortest solutions, and how close were they to the known optimum?**

The known optimum from BFS was 3 moves for every puzzle, but since no model solved a cube, no optimality gap could be calculated. Looking at the moves, the fast models mostly repeated the same move or switched between a move and its opposite (for example Jev kept playing U and U’, and Gemma played U’ over and over), so they almost never got closer to the solved state. The only run that ended closer was Qwen3.5 on ollama-3 (from 22 to 29 correct stickers out of 54), and it still did not solve the cube.

**4. Did the fastest model also achieve the best solution quality?**

No. The fastest models (Clef on OpenRouter, Gemma3 on Ollama) answered quickly, but their moves did not move the cube toward the solution. In this experiment, being fast did not mean making good decisions: speed without planning just produced loops.

**5. Would you use the fastest decision model, the frontier LLM, or a hybrid agent in a real-world application? What evidence supports your choice?**

I would use a hybrid agent. My results show that the fast models alone could not solve even a 3 move cube: they were cheap and quick, but got stuck repeating moves. A frontier LLM is expensive and, as this experiment showed, limited by API calls and credit, so it can’t be called for every small step. A better design is to let a reliable tool (like the BFS solver I already use to find the optimum) handle the hard planning, use a cheap fast model for simple routine steps, and only call the expensive LLM when the agent is stuck. This keeps quality high while keeping cost and latency low.

Limitations: only one OpenRouter puzzle because of the API call limits; two OpenRouter models could not make valid moves; the Ollama models ran on a laptop CPU, so their latency is much higher than on a GPU, and Clef Flash was too big for the laptop’s memory, so none of its calls succeeded.

## Project structure

```
rubiks-cube-race/
  backend/
    main.py               FastAPI app: API routes, race and replay WebSockets, serves the frontend
    config.py             settings (move limit, timeouts, model IDs), reads .env
    cube_engine.py        cube state, the 18 moves, scramble, is_solved
    solver_bfs.py         bidirectional BFS for the shortest solution
    puzzles.py            makes and loads the fixed test sets
    benchmark.py          runs the race, retries, logging, saving, ranking, dashboard stats
    api_guard.py          OpenRouter safety: request limits, persistent counter, response cache
    adapters/
      base.py             common interface, shared prompt wording, move validation, HTTP call
      mock_adapter.py     mock-random, mock-oracle, mock-noisy, mock-broken
      ollama_adapter.py   the four local Ollama models (chat for the LLMs, decision endpoint for Clef)
      decisions_adapter.py   Jev, Luna and Clef (OpenRouter Decisions API)
      chat_adapter.py        Opus (OpenRouter chat completions)
  frontend/
    index.html            the one page: race screen and results below it
    style.css             all styles
    common.js             small shared helpers
    app.js                race screen logic (WebSocket, panels, cube drawing, timers)
    dashboard.js          results section logic (tables, Chart.js charts, slide mode)
  data/puzzles.json       the saved test puzzles
  data/api_usage.json     OpenRouter request counter (created by the first real request, never reset)
  data/api_cache.json     saved OpenRouter responses (created by the first real request)
  results/                one JSON file per model per race
  tests/                  pytest tests (conftest.py blocks the network and keeps tests away from data/)
  .env.example            template for the API key
  requirements.txt
  README.md
```
