from openai import OpenAI
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# Methodology and results get more tokens — all others are tight
SECTION_MAX_TOKENS = {
    "topic":             200,
    "motivation":        200,
    "literature_review": 250,
    "dataset":           400,
    "methodology":       700,
    "results":           700,
    "insights":          200,
    "limitations":       180,
    "future_research":   220,
}

SECTION_PROMPTS = {

"topic": """
Read the paper and answer in exactly 4-5 sentences:
1. What is this paper about? (1 sentence)
2. What specific problem does it solve? (1-2 sentences)
3. What is the proposed solution in brief? (1 sentence)
4. What field/domain? (1 sentence)

No history. No background. Direct answers only.

PAPER: {text}
""",

"motivation": """
Read the paper. Answer in 4-5 sentences only:
1. What problem or gap does this paper address?
2. Why did existing approaches fail or fall short?
3. What does this paper aim to fix?

Facts from the paper only. No generic filler.

PAPER: {text}
""",

"literature_review": """
List ONLY the key prior works cited in this paper.
Format: bullet points.
Each bullet = work name/authors + what it did + what method they adapted to fulfil the research+
what results they achieved(mentioning the metrics they used to define their best results)+ its limitation.
Max 4-5 works. End with 1 sentence on how this paper differs.
Total under 120 words.

PAPER: {text}
""",

"dataset": """
You are an expert researcher  specializing in analyzing research papers.

Your task is to carefully read the provided research paper and extract **ALL information related to the dataset and data collection process**. Do not miss any detail. Be precise, structured, and comprehensive.

Follow the instructions strictly:

1. DATASET IDENTIFICATION
- Name of the dataset(s) used
- Whether the dataset is:
  a) Public (existing benchmark dataset)
  b) Private (collected by authors)
  c) Combination of multiple datasets

2. DATASET SOURCE
- If public: mention source (e.g., Kaggle, UCI, custom benchmark, etc.)
- If private (created by authors): clearly state it

3. DATA COLLECTION PROCESS (VERY IMPORTANT — DO NOT MISS ANY DETAIL)
If the dataset is created by the authors, extract and explain:
- How the data was collected:
  • Web scraping / crawling (mention websites if available)
  • APIs used
  • Manual collection
  • Sensors / devices / experiments
  • Surveys / annotations
- Step-by-step pipeline of data collection (if described)
- Any tools, scripts, or frameworks used
- Time period of data collection
- Criteria for selecting data
- Ethical considerations (if mentioned)

4. DATASET STRUCTURE & CONTENT
Extract complete details about:
- Total number of samples/data points
- Number of classes/labels (if applicable)
- Class distribution (balanced/imbalanced + numbers if given)
- Number of features/columns
- Description of each column/feature (if available)
- Type of data:
  • Text / Image / Audio / Video / Multimodal / Tabular
- File formats (e.g., CSV, JSON, images, etc.)

5. DATA PREPROCESSING & CLEANING
Extract ALL preprocessing steps:
- Data cleaning (removal of noise, duplicates, missing values)
- Text preprocessing (tokenization, stopword removal, stemming, etc.)
- Image preprocessing (resizing, normalization, augmentation, etc.)
- Audio preprocessing (sampling rate, spectrogram, etc.)
- Feature engineering (if any)
- Label encoding or annotation process
- Train/validation/test split details

6. DATA AUGMENTATION (if any)
- Techniques used
- Why they were used

7. DATASET CHALLENGES / LIMITATIONS
- Any issues mentioned:
  • Bias
  • Imbalance
  • Noise
  • Small dataset size

8. OUTPUT FORMAT
- Present the answer in a clean structured format with headings
- Use bullet points where appropriate
- Do NOT hallucinate — only extract what is explicitly mentioned

IMPORTANT:
- Do not skip even minor dataset-related details
- Be exhaustive and precise
- Focus ONLY on dataset and data-related aspects (ignore model unless related to preprocessing).
*** Only include information explicitly supported or clearly inferable from the paper; omit any sections or points that are not addressed, without mentioning their absence.
PAPER: {text}
""",

"methodology": """
Carefully read the provided research paper and extract a COMPLETE and DETAILED explanation of the METHODOLOGY used in the study. 


Instructions:

1. Identify all sections related to methodology (e.g., "Methodology", "Proposed Method", "Approach", "Framework", "Model Architecture", "Experimental Setup").

2. Extract and clearly explain the methodology in a structured and comprehensive manner. Your explanation MUST include:

   A. Overall Approach
   - What problem is being solved?
   - What type of approach is used (e.g., supervised learning, multimodal learning, transformer-based, etc.)?

   B. Data Handling
   - Dataset(s) used
   - Data preprocessing steps
   - Data splitting (train/validation/test)
   - Any augmentation or balancing techniques

   C. Model Architecture / System Design
   - All models/components used (e.g., BERT, CNN, CLIP, etc.)
   - Architecture details (layers, modules, flow of data)
   - If multimodal: explain how different modalities are processed and fused

   D. Feature Extraction & Representation
   - How features are extracted (text/image/audio)
   - Embedding techniques or encoders used

   E. Fusion Strategy (if applicable)
   - Early fusion / Late fusion / Cross-modal attention
   - Detailed explanation of how features are combined

   F. Training Procedure
   - Loss functions
   - Optimization algorithm (e.g., Adam, SGD)
   - Hyperparameters (learning rate, batch size, epochs)
   - Regularization techniques (dropout, freezing layers, etc.)

   G. Evaluation Setup
   - Metrics used (accuracy, F1-score, etc.)
   - Validation strategy (cross-validation, holdout)

   H. Mathematical Formulation (if present)
   - Extract and explain all important equations in simple terms

3. Do NOT skip any important methodological detail, even if it seems minor.

4. If the paper contains diagrams or figures (for VLMs), interpret them and include their methodological meaning.

5. Rewrite everything in SIMPLE, CLEAR, and WELL-STRUCTURED language so that a graduate-level student can understand it easily.

6. If any part of the methodology is unclear or missing, explicitly mention it.

7. Format the output using clear headings and bullet points.

Output Format:

- Title: Methodology Explanation
- Sections: A → H (as listed above)
- Use bullet points + short paragraphs
- Keep it detailed but organized.
*** Only include information explicitly supported or clearly inferable from the paper; omit any sections or points that are not addressed, without mentioning their absence.
PAPER: {text}
""",

"results": """
You are an expert AI research analyst with deep knowledge of machine learning, NLP, computer vision, and multimodal systems.

Your task is to carefully read the provided research paper and extract a COMPLETE and STRUCTURED analysis focused ONLY on the RESULTS and EVALUATION aspects.

Follow these instructions strictly:

1. RESULTS SUMMARY
- Clearly state the main results achieved in the paper.
- Include numerical performance (e.g., accuracy, F1-score, BLEU, AUC, etc.).
- Mention improvements over baselines or prior work.
- Highlight any state-of-the-art (SOTA) claims.

2. ⚙️ HOW THE RESULTS ARE ACHIEVED
- Explain the methodology or key techniques that led to the results.
- Identify specific components responsible for improvements (e.g., attention mechanism, fusion strategy, loss function, architecture design).
- If ablation studies are present, explain which components contributed most and why.

3. EVALUATION METRICS
- List all metrics used in the paper.
- Define each metric briefly (mathematical intuition if relevant).
- Explain WHY each metric is appropriate for the task (e.g., why F1-score for imbalanced data, why BLEU for text generation, etc.).

4. EXPERIMENTAL SETUP (RESULT-RELATED ONLY)
- Mention datasets used for evaluation.
- Describe train/test split, cross-validation, or benchmarking strategy (if relevant to results).
- Note any important evaluation protocols (e.g., zero-shot, few-shot, cross-domain).

5. 📈 RESULT ANALYSIS & INTERPRETATION
- Interpret what the results actually mean.
- Explain strengths and weaknesses revealed by the results.
- Identify any overfitting/underfitting signals.
- Discuss generalization ability if mentioned.

6. 🔍 COMPARISON WITH BASELINES
- List baseline models/methods used for comparison.
- Explain how the proposed method outperforms or underperforms them.
- Provide reasoning behind performance differences (not just numbers).

7. LIMITATIONS IN RESULTS
- Extract any limitations, inconsistencies, or failure cases mentioned.
- Identify missing evaluations or potential biases.

8. 💡 CRITICAL INSIGHT (IMPORTANT)
- Go beyond the paper: infer WHY the method works based on results.
- Identify the core idea that most likely drove performance gains.
- If possible, relate findings to broader trends in AI (e.g., multimodal fusion, transformer scaling).

OUTPUT FORMAT:
- Use clear headings for each section.
- Use bullet points for clarity.
- Be precise, technical, and avoid vague statements.
- Do NOT summarize the entire paper — focus ONLY on results and evaluation.

If tables/figures are present:
- Extract and interpret key findings from them.
- Do not just restate values — explain their significance.

If the paper involves multimodal learning:
- Pay special attention to multimodality(integration of modalities) and their impact on results.

Your explanation should be detailed enough that a researcher can understand HOW and WHY the model achieved its performance — not just WHAT it achieved.
*** Only include information explicitly supported or clearly inferable from the paper; omit any sections or points that are not addressed, without mentioning their absence.
PAPER: {text}
""",

"insights": """
Give exactly 3 key insights from this paper as bullet points.
Each bullet: 1-2 sentences — what the insight is + why it matters specifically for this paper.
No generic observations. Only insights specific to this paper's findings.

PAPER: {text}
""",

"limitations": """
List limitations of this paper as bullet points.
- First: limitations stated by the authors (quote or paraphrase directly)
- Then: 1-2 limitations you identify that authors did not mention

Under 100 words total. No padding.

PAPER: {text}
""",

"future_research": """
Suggest 3-4 future research directions as bullet points.
Each: direction name + 1 sentence on why it matters for this specific work.
Include at least one idea on how LLMs could extend this research.
All ideas must connect directly to this paper's topic and limitations.
Under 130 words.

PAPER: {text}
"""
}


def tutor_explain_section(section_key: str, text: str) -> str:
    if section_key not in SECTION_PROMPTS:
        raise ValueError(f"Unknown section: {section_key}")

    prompt = SECTION_PROMPTS[section_key].format(text=text[:4000])
    max_tok = SECTION_MAX_TOKENS.get(section_key, 300)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a sharp, concise research explainer. "
                    "Every sentence must carry a specific fact from the paper. "
                    "Never add background history, never pad, never repeat. "
                    "Strictly follow the word/sentence limits. "
                    "If information is not in the paper, skip that point silently."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=max_tok
    )

    return response.choices[0].message.content.strip()


def tutor_explain_full(full_text: str) -> dict:
    sections_order = [
        "topic", "motivation", "literature_review", "dataset",
        "methodology", "results", "insights", "limitations", "future_research"
    ]
    explanations = {}
    for section in sections_order:
        try:
            explanations[section] = tutor_explain_section(section, full_text)
        except Exception as e:
            explanations[section] = f"Could not generate explanation: {str(e)}"
    return explanations


def tutor_explain(text: str) -> str:
    return tutor_explain_section("topic", text)