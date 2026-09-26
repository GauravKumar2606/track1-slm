# Track 1 SLM — Model Comparison Report

Models evaluated: Model A (English-only, from scratch, base pretrain), Model A-Instruct (English base + instruction fine-tune), Model B (English+C#+instructions, from scratch), Model C (fine-tuned from Model A-Instruct on C#-only)

## Perplexity Matrix

Rows = model that generated the checkpoint being scored; columns = which model's test set it was scored against. The diagonal is each model's perplexity on its own held-out data.

| Model \ Test set | A | A-INSTRUCT | B | C |
|---|---|---|---|---|
| **A** | 48.75 (own) | 14.71 | 46.05 | 16.38 |
| **A-INSTRUCT** | 63.44 | 6.31 (own) | 61.37 | 15.34 |
| **B** | 50.08 | 8.33 | 47.03 (own) | 7.23 |
| **C** | 86.02 | 10.30 | 80.45 | 5.51 (own) |


## Side-by-Side Responses

| Question | Model A (English-only, from scratch, base pretrain) | Model A-Instruct (English base + instruction fine-tune) | Model B (English+C#+instructions, from scratch) | Model C (fine-tuned from Model A-Instruct on C#-only) |
|---|---|---|---|---|
| What is Python? | What is Python? | What is Python? This is a language that is designed to be used in a variety of applications, including real-time data processing and communication. It has some similarities with Node!js which allows you to send HTTP requests like this: ```python import n | What is Python? It's a simple, intuitive process. | What is Python? Python, for example - not a module to use /// If the argument should've been "created", then we can't create the . // However, if the arguments cannot have passed in any way, they will be created. Python |
| Write a C# for loop | Write a C# for loop, and you can create the default to write, but with this project it now has about 100 lines of code. | Write a C# for loop. Here is an example of how you can implement the `cpp-nn` macro: #include using namespace std; int main() { int cp_mode = "movie"; int xml_decode | Write a C# for loop ### Response: $$ A_1 = 5 | Write a C# for loop. public static void Start() { Console.WriteLine("Started"); } /// Checks if the loop is already started. (see Debugger) private static bool IsInitialized { get; set; } = |
| What is Machine Learning? | What is Machine Learning? | What is Machine Learning? The process of machine learning involves the exploration and evaluation of human behavior, such as problem-solving capabilities, decision trees, and general relativity. This requires understanding the various layers of cognitive function that can be used to model this behavior. A machine | What is Machine Learning? | What is Machine Learning? public static string Machine Learning = ""; /// A list of machine learning algorithms to train. This allows for a dataset collection that includes the following features: classifier, trainingData, algorithm1 }; private List>> ListSelencersTo |
| What is AGI? | What is AGI? | What is AGI? A. The main principles of Caffeine and its associated mechanisms for regulating metabolism have been explained through various molecular mechanisms, including: 1. Regulation of metabolic pathways: These pathways are involved in the regulation of metabolic processes within a organism's cells, influencing | What is AGI? (G) "Currently, my experience with ACP is that you have a fairly long process of development time and then some to work on. If it would be easier for me to do it in time, I will just be working at | What is AGI? What is the purpose of this search result?"); } [TestMethod] public void GetBroadcaster_Should_ReturnsValidData([NotNullWhen(false))] string data) { |
| Explain gravity | Explain gravity and the impact of an extreme event. | Explain gravity as the relationship between the motion of a planet and its gravitational pull. When a planet is at rest, it pulls more easily off than on Earth to reach its target orbital (the center of all surface elements), which results in an increase in force | Explain gravity. | Explain gravity and move the player to it. /// public static float GetVelocity(double v) { return GetRadius(v); } #endregion } |
| What is the capital of France? | What is the capital of France? What are our foreign policy objectives, which we know to be hostile or ineffective?" | What is the capital of France? The city where he was born, also known as Paris. It's a country located in France that lies between the French and English lands. The name may indicate his birthplace or personal location, but it can still be used for an additional purpose | What is the capital of France? The | What is the capital of France? |
| Tell me a short story | Tell me a short story of this game. In the book, there are two main characters who use their character's skills in order to accomplish various goals and objectives, but each of them uses its own abilities as well: | Tell me a short story about a family who have survived the long winter and need to rebuild their home in order to return to their hometown. The house was built by an architect named Jethro A. Tzu, who had lived for over two decades after being in | Tell me a short story about a guy who lived all this time in America. The man made it to New York, and he got the job back later that same day I had an interview with him: | Tell me a short story for the app developer guide.")] public partial class AndroidAppBuilder { const string ApplicationName = "roidapp"; using (var context = new AndroidScriptContext()) { var appUri = GetApplicationUri(App |
| What is a neural network? | What is a neural network? The brain has been shown to be able to do so for many years now in the same way that MRI can, in the very limited sense that it does. | What is a neural network? This question asks about learning behavior in the brain and how it affects our cognitive processes. The question focuses on the relationship between learning rate, memory, and self-awareness. Here are some ways to answer this question: 1. Learning rate: One step | What is a neural network? | What is a neural network? /// public static class Dynagram { #region Fields /// Gets the name of the neuron. /// /// The neuron to be searched. /// public string Name; /// /// Gets or sets a |


## Key Observations

- Model A (English-only, from scratch, base pretrain) scores 48.75 perplexity on its own test set.
- Model A-Instruct (English base + instruction fine-tune) scores 6.31 perplexity on its own test set.
- Model B (English+C#+instructions, from scratch) scores 47.03 perplexity on its own test set.
- Model C (fine-tuned from Model A-Instruct on C#-only) scores 5.51 perplexity on its own test set.
- A performs worse on A-INSTRUCT's test set (14.71) than A-INSTRUCT does on its own (6.31) — A-INSTRUCT's data is out-of-distribution for A.
- A-INSTRUCT performs worse on A's test set (63.44) than A does on its own (48.75) — A's data is out-of-distribution for A-INSTRUCT.
- A performs worse on C's test set (16.38) than C does on its own (5.51) — C's data is out-of-distribution for A.
- C performs worse on A's test set (86.02) than A does on its own (48.75) — A's data is out-of-distribution for C.
- A-INSTRUCT performs worse on B's test set (61.37) than B does on its own (47.03) — B's data is out-of-distribution for A-INSTRUCT.
- B performs worse on A-INSTRUCT's test set (8.33) than A-INSTRUCT does on its own (6.31) — A-INSTRUCT's data is out-of-distribution for B.
- A-INSTRUCT performs worse on C's test set (15.34) than C does on its own (5.51) — C's data is out-of-distribution for A-INSTRUCT.
- C performs worse on A-INSTRUCT's test set (10.30) than A-INSTRUCT does on its own (6.31) — A-INSTRUCT's data is out-of-distribution for C.
- B performs worse on C's test set (7.23) than C does on its own (5.51) — C's data is out-of-distribution for B.
- C performs worse on B's test set (80.45) than B does on its own (47.03) — B's data is out-of-distribution for C.


## Conclusion

- **A vs A-Instruct** — the foundational comparison: identical base pretrain, only the instruction fine-tuning stage differs. A clean, single-variable comparison — isolates exactly what instruction-tuning changed (and cost, if anything) relative to the untouched base model.
- **A vs B** — the comparison this project originally started with: same from-scratch training procedure and token budget, only the data mix differs (English-only vs. English+C#+instructions). A clean, single-variable comparison — isolates the effect of adding C#+instructions to the pretraining mix, holding everything else constant.
- **A vs C** — since C's lineage runs through A-Instruct back to A, this comparison spans two sequential fine-tuning stages (instruction-tuning, then C#) relative to the original base pretrain. Not single-variable (two stages changed at once), but shows the cumulative drift from the common ancestor both A-Instruct and C share.
- **A-Instruct vs B** — NOT a clean single-variable comparison (known, accepted limitation): B differs from A-Instruct in two ways at once — joint-from-scratch training procedure AND a different overall data mix — so any difference can't be cleanly attributed to just one cause.
- **A-Instruct vs C** — since C is initialized from A-Instruct's own weights, this shows what fine-tuning on C# changed relative to the exact checkpoint it started from (catastrophic forgetting of instruction-following vs. C# specialization).
- **B vs C** — a clean, meaningful comparison: both models end up exposed to C#+instructions, but B learned everything jointly from scratch while C learned English+instructions first, then C# sequentially via fine-tuning. Differences here isolate *how* it was learned (joint vs. sequential), independent of the A-Instruct-vs-B caveat above.
- **A vs A-INSTRUCT vs B vs C** — the full picture across every trained model. Single-variable, directly interpretable: A vs A-INSTRUCT, A vs B, B vs C. Two-variable, interpret with caution: A vs C, A-INSTRUCT vs B. 
