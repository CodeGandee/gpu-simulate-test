In the **Vidur** MLSys’24 paper, their “real experiments” baseline is described as **an optimized fork of vLLM**: they say they run experiments on “an optimized version of the vLLM codebase” and that they “use a fork of the open-source implementation of vLLM” extended with **extra scheduling policies, chunked prefills, and telemetry** (and they also mention CUDA graphs). ([arXiv][1])

**Where is that modified vLLM?**

* It is **not in** `microsoft/vidur` (that repo is the simulator). The paper also doesn’t give a link to a dedicated public “vLLM fork” repo—its references point to the upstream vLLM GitHub. ([arXiv][1])
* The closest (and likely what you’re looking for in practice) is **Microsoft’s `microsoft/sarathi-serve`**, which is explicitly a **fork-derived codebase from vLLM** (“This repository originally started as a fork of the vLLM project”) and is **public/open-source (Apache-2.0)**. ([GitHub][2])
  Microsoft Research also links Sarathi-Serve as the available source code for that line of work. ([Microsoft][3])

**So:**

* **Exact Vidur paper “optimized vLLM fork”**: not clearly published as a standalone vLLM fork repo.
* **Open-source codebase with the vLLM-derived modifications (chunked-prefills / scheduling research)**: `microsoft/sarathi-serve` is public. ([GitHub][2])

[1]: https://arxiv.org/pdf/2405.05465 "Vidur: A Large-Scale Simulation Framework for LLM Inference"
[2]: https://github.com/microsoft/sarathi-serve "GitHub - microsoft/sarathi-serve: A low-latency & high-throughput serving engine for LLMs"
[3]: https://www.microsoft.com/en-us/research/publication/taming-throughput-latency-tradeoff-in-llm-inference-with-sarathi-serve/ "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve - Microsoft Research"

---

look into github issues, for people who want to replicate vidur's result, how do they pick the "real" inference engine?

From the repo’s **issues**, people who try to “replicate Vidur” often run into exactly this ambiguity: they benchmark **upstream vLLM** and see huge latency mismatches, then ask **which “real” engine Vidur expects** (vLLM vs Sarathi-Serve, and which version). ([GitHub][1])

What the **open-sourced Vidur artifact actually assumes today** is:

### The “real” engine for Vidur’s profiling path is **Sarathi-Serve**

* Vidur’s README examples default to the **Sarathi** scheduler (`--replica_scheduler_config_type sarathi` with chunk/batch caps), which is the policy they’re modeling/running in most “happy path” examples. ([GitHub][2])
* Users digging into the code path noticed (and ask about) the same thing: “Vidur’s profiling is based on Sarathi-Serve”. ([GitHub][3])
* And you can see it very concretely in the Sarathi-Serve issue raised by someone running Vidur’s profiling scripts: the traceback shows Vidur’s profiling calling into **`sarathi-serve/sarathi/model_executor/...`** (i.e., Sarathi-Serve code is in the execution path). ([GitHub][4])

So, when people want an apples-to-apples **“real vs simulated”** validation that matches the **released** Vidur setup, the closest “real inference engine” to use is **Sarathi-Serve**, not a random upstream vLLM version.

### What about vLLM?

Vidur *can* simulate a “vLLM v1” style scheduler (you’ll see configs like `replica_scheduler_config_type vllm_v1` mentioned in replication questions), but the repo does **not** ship a turn-key “run this exact vLLM fork/version and compare” harness—hence why issue #60 exists. ([GitHub][1])

**Practical takeaway for replication:**

* If you’re trying to reproduce the **open-source Vidur** numbers/behavior, use **Sarathi-Serve + Vidur** (Sarathi for profiling + Sarathi scheduler in simulation). ([GitHub][2])
* If you specifically want **vLLM** as the “real engine,” you’ll need to pin a vLLM version + scheduler settings yourself and accept that it may not match Vidur’s released profiling/scheduler assumptions (which are Sarathi-centric). ([GitHub][1])

[1]: https://github.com/microsoft/vidur/issues/60 "Current vidur backend support · Issue #60 · microsoft/vidur · GitHub"
[2]: https://github.com/microsoft/vidur "GitHub - microsoft/vidur: A large-scale simulation framework for LLM inference"
[3]: https://github.com/microsoft/vidur/issues/51 "Adapting Vidur to vLLM and Profiling CPU Overhead · Issue #51 · microsoft/vidur · GitHub"
[4]: https://github.com/microsoft/sarathi-serve/issues/55 "Error when using Vidur profiling for Llama-2-70b attention core · Issue #55 · microsoft/sarathi-serve · GitHub"

---

I dug through the **public Vidur GitHub issues** and also looked for **external repos/papers** that mention running Vidur.

### What I see in Vidur’s GitHub issues

I **did not find a public issue thread where someone explicitly reports “I successfully reproduced the Vidur paper’s sim-vs-real numbers/plots”** (or “<9% error” fidelity) end-to-end. Instead, the replication-related issues that are visible are mostly **“can’t match / big mismatch / which engine is the reference?”**:

* **Issue #60**: a user says their “real” benchmark using **vLLM 0.9.1** differs hugely (they mention **~700% P50 latency difference**) and asks what engine/version Vidur expects for reproducing published results, and whether Vidur is based on vLLM or Sarathi-Serve. ([GitHub][1])
* **Issue #72**: a user tried running Vidur’s profiling workflow on **AWS H100 (P5 48x)** and reports the collected profiling data differs significantly from the repo’s provided profiling CSVs (flashinfer vs flash_attention, extra columns, and prediction changes). ([GitHub][2])
* **Issue #51**: another user highlights that Vidur’s profiling is based on **Sarathi-Serve** and asks how to adapt properly to **vLLM** (implying “just switching scheduler flags” won’t match latest vLLM behavior). ([GitHub][3])

So, based on the public issues, the common story is **“people are trying, but they’re not matching out-of-the-box.”**

### Evidence that others *can run Vidur* (but not necessarily “reproduce Vidur paper plots”)

Even though I don’t see public “matched-the-paper” confirmations, there *is* evidence Vidur is getting used in other work:

* **Derivative repos** exist (e.g., a project extending Vidur to add sustainability/energy metrics). ([GitHub][4])
* Some papers explicitly say they **evaluate in Vidur** (using it as a simulator in their experiments). ([MIT CSAIL][5])
* The **MaverIQ** SC’25 paper includes an **artifact appendix** that instructs users to clone/install **Vidur + Sarathi-Serve** and references baseline scripts, which suggests they had a working, reproducible baseline setup that includes Vidur. ([ut-sysml.ece.utexas.edu][6])

### Bottom line

* **Publicly visible**: I don’t see someone saying “yes, I reproduced Vidur’s published sim-vs-real results exactly.”
* **But**: Vidur *is* being used downstream (and embedded in other artifacts), so “it runs / it’s usable” is clearly true; it’s the **exact replication of Vidur’s headline fidelity numbers** that doesn’t show up as a clean success story in the issues. ([GitHub][1])

[1]: https://github.com/microsoft/vidur/issues/60 "Current vidur backend support · Issue #60 · microsoft/vidur · GitHub"
[2]: https://github.com/microsoft/vidur/issues/72 "Potential Issue Regarding Profiling · Issue #72 · microsoft/vidur · GitHub"
[3]: https://github.com/microsoft/vidur/issues/51 "Adapting Vidur to vLLM and Profiling CPU Overhead · Issue #51 · microsoft/vidur · GitHub"
[4]: https://github.com/ozcanmiraay/vidur-energy?utm_source=chatgpt.com "ozcanmiraay/vidur-energy"
[5]: https://people.csail.mit.edu/pkarimib/papers/Glia.pdf?utm_source=chatgpt.com "Glia: A Human-Inspired AI for Automated Systems Design and ..."
[6]: https://ut-sysml.ece.utexas.edu/publications/prints/sc2025_liakopoulos.pdf?utm_source=chatgpt.com "MaverIQ: Fingerprint-Guided Extrapolation and Fragmentation ..."
