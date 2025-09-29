### Model overview

The model is a fine-tuned RoBERTa-base transformer for emotion classification, adapted from the pre-trained checkpoint Jorgeutd/sagemaker-roberta-base-emotion. It is designed to predict emotion labels (surprise, disgust, neutrality, fear, sadness, happiness, anger) from text inputs. The model leverages transfer learning by retraining the base RoBERTa architecture on a combination of the dair-ai/emotion dataset and custom test/validation sets generated through a collaborative human annotation process and internal pipeline provided by the Content Inteligence Agency (https://www.contentintelligence.nl/).

Dair-ai dataset: https://github.com/dair-ai/emotion_dataset/blob/master/README.md

Model used: https://huggingface.co/Jorgeutd/sagemaker-roberta-base-emotion


### Architecture


Base Model: roberta-base (125M parameters, 12 transformer layers and 12 attention heads, 768 hidden dimensions).

Tokenizer: RoBERTa tokenizer with a sequence length of 128 tokens (max_length=128), using padding and truncation.

Classification Head: A custom sequence classification layer with num_labels corresponding to the emotion classes in the dataset.

Training Framework: Hugging Face Trainer API with PyTorch.

Hyperparameters:

Epochs: 10

Batch Sizes: 16 (train), 32 (eval)

Learning Rate: 3e-6 (fine-tuning rate)

Optimizer: AdamW (default for Trainer)

Task-Specific Modifications:

Classification Head:

Pooler: Uses RoBERTa’s [CLS] token embedding (dimension 768) as the sequence representation.

Dropout: Applied after pooling (default hidden_dropout_prob=0.1).

Linear Layer: Maps pooled output to 7 logits (one per emotion class).

Label Adaptation:

Original pretrained model (Jorgeutd/sagemaker-roberta-base-emotion) had 6 labels.

Modification: Added a 7th output neuron for the neutral class using ignore_mismatched_sizes=True to retain pretrained weights.

### Purpose

The model was developed to:

Classify text into 7 emotion categories, expanding beyond the original 6-class dair-ai/emotion dataset.

Adapt to domain-specific data (e.g., product reviews, support tickets) while retaining performance on general social media text.

Improve robustness using weighted F1-scores and confusion matrixes as a key metrics.

### Development Context

Hardware: Trained on 4× NVIDIA L40S GPUs (primary GPU utilized 98%, 33GB VRAM).

Training Time: ~25 minutes for 10 epochs.

Datasets:

Primary: dair-ai/emotion (16k train, 2k validation/test).

Custom:

val_df_all.csv: Internal domain-specific data (16k train, 5.5k validation/test). Size: 16562 train / 5521 val / 5521 test (used for validation/testing)

test_df_all.csv: Student-annotated diverse text (~800 samples). Size: 2391 train / 797 val / 798 test

Label Mapping: Extended original 6-class labels to 7 classes (added neutral).

### Key Assumptions

Pre-trained roberta-base (from Jorgeutd/sagemaker-roberta-base-emotion) provides a strong foundation for transfer learning on emotion classification.

Domain-specific data (e.g., internal pipeline data) and student-annotated texts reflect real-world scenarios not fully captured by the social media-focused dair-ai dataset.

It assumes that emotion labels are mutually exclusive; mixed-emotion texts may lead to poor predictions.

### Constraints

Training Data Size: Limited to 16,000 samples from dair-ai/emotion and ~16,562 custom training samples (from val_df_all.csv and test_df_all.csv).

Hardware: Trained on NVIDIA L40S GPUs (4 GPUs, but only GPU 0 was heavily utilized at 98% compute, 33.3GB/46GB VRAM).

Training Time: 10 epochs took 25 minutes (25:21 runtime), constrained by GPU memory and batch size settings.

### Performance:

Training Loss: 0.5636

Weighted F1 Score: 0.6828

Validation Accuracy: 69.18%

Confusion matrix: 

![alt text](image-2.png)

![image.png](image.png)

Data distribution:

![alt text](image-1.png)

### Intended use

Emotion classification models like the one trained in this project can be applied across diverse real-world domains:

- Detect emotions in user comments, tweets, or reviews to gauge public sentiment about brands, products, or events.
Example: Identifying spikes in "anger" or "fear" in customer feedback during a PR crisis on Twitter.
- Analyze support chat transcripts or survey responses to categorize emotions such as "frustration" in complaints or "joy" in positive reviews.
Example: A media company analyzing viewer reactions to a newly launched TV series to detect prevailing sentiments.
- Evaluate emotions in video transcripts, podcasts, or articles to assess tone and audience engagement.
Example: Measuring the presence of "surprise" and "joy" in YouTube video transcripts to optimize storytelling strategies for entertainment platforms.
- Move beyond basic sentiment (positive, negative, neutral) to identify more nuanced emotional states like "love", "fear", or "sadness" that better inform brand positioning or content tone.
- Understand the emotional tone in customer interactions, especially through email or chat logs, to improve service strategies.
Example: Automatically flagging messages expressing "frustration" to escalate support requests more efficiently.
- Apply emotion detection on articles, books, or narratives to study audience impact, bias, or emotional arcs in storytelling.
- Assist in early detection of emotional distress by analyzing communication in texts, journals, or online forums.
Example: Monitoring signs of "sadness" or "fear" in user-submitted messages on mental health platforms.

Limitations and Context avoidance:

- No ethical safeguards are built-in to handle sensitive or ambiguous text as it has a semnificative risk for misinterpretation.
- Tokenizer (RobertaTokenizer) and pretraining/fine-tunning data lack multilingual, dialectal coverage  and is incompatible with non-Latin scripts.
- Struggles with narratives where emotional cues span multiple sentences and non-standard grammar is used.
- Trained primarily on social media text (dair-ai/emotion) and video content transcripts.
- The data used lacks explicit handling of sarcasm, irony, or cultural nuances.
- English-only training data limits global applicability.
- Reflects biases of Western social media and video content.
- Model cannot generalize to non-English languages due to training on English-only data.
- Should not be used in a medical context to identify any level of distress or as an aid to a diagnostic

Other limitations after studying the translation: https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group_26_y2c/blob/main/datalab_tasks/Task6/Translation_error_analysis.pdf

### Performance Metrics and Evaluation

The analysis aims to identify where the emotion classification model fails and why, using confusion matrices, misclassifications, and performance metrics. The dataset is imbalanced, with "neutral" (440) and "happy" (329) dominating, while "disgusted" (17) and "scared" (16) are rare, leading to biased predictions. The confusion matrix shows strong accuracy for "neutral" (343 correct) and "happy" (230), but frequent confusion between "neutral" and "happy," and misclassifications like "happy" → "surprised," "surprised" → "neutral," and frequent errors on "scared," "mad," and "disgusted." Misclassifications often involve ambiguous or metaphorical language, with some cases suggesting inconsistencies in test set labeling. Common errors include "happy" → "neutral" (53), "neutral" → "happy" (40), and "surprised" → "neutral" (24), showing emotional overlap and weak cues. Misclassified texts are longer (7.05 tokens) than correct ones (5.70), suggesting complexity increases difficulty. Performance is strong for "neutral" (F1 = 0.78) and "happy" (F1 = 0.73), weak for "disgusted," "scared," and "sad," with overall accuracy at 69%. Emotion detection is challenging due to missing tone, subjective language, and overlapping emotions. To improve: balance the dataset, use contrastive learning, explore multimodal models, and refine test set labeling. In sum, most errors stem from imbalance, ambiguity, and subtle expressions—better data and models can improve emotion detection.


Please check this link for a thorough and complete analysis: https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group_26_y2c/blob/main/Error%20Analysis.pdf



### Explainability and Transparency: 

- Misinterpretation of Emotionally Charged Words

Example: "Revolting" might decrease the emotional weight of "mad," or "happiness" could decrease "happy."

Explanation: The model misjudges emotionally significant words by assigning counterintuitive relevance scores, leading to incorrect emotional classifications.

- Tokenization Errors Distort Meaning

Example: Splitting "tears" into "T" and "ears" weakens the connection to sadness.

Explanation: When words are broken down improperly, the model loses context, making it harder to understand the full emotional meaning.

- Over-Reliance on Specific Tokens

Example: Emotion prediction depends heavily on sentence-ending punctuation (e.g., "!" or </s>).

Explanation: The model focuses too much on specific tokens like punctuation, missing the broader emotional context.

- Confidence Drops When Critical Tokens Are Removed

Example: Removing words like "astonished" or "frustrating" leads to sharp drops in confidence.

Explanation: The model depends too much on individual words, and removing key terms can make predictions less reliable.

- Inconsistent Explanations Across XAI Methods

Example: "Wow!" is interpreted differently by Gradient × Input and Layer-wise Relevance Propagation (LRP), where one deems it negative and the other neutral.

Explanation: Different explanation methods provide conflicting insights, making it harder to trust or refine the model’s reasoning.

- Layer-Wise Noise in XAI

Example: LRP spreads relevance too evenly, diluting focus on key emotional cues.

Explanation: Some XAI techniques distribute relevance poorly across layers, failing to pinpoint significant emotional indicators.

- Struggles with Modifier Words

Example: Words like "completely" or "so" alter the intensity of emotions but are treated in isolation.

Explanation: The model doesn’t fully account for words that modify or amplify emotion, which impacts accuracy in emotion detection.

- Failure to Capture Word Interplay

Example: "Rotten tomatoes" isn’t recognized as a disgust-inducing phrase.

Explanation: The model struggles with phrases that should be understood as a whole, misinterpreting individual word meanings.

- Bias from Training Data

Example: "Healthcare" may be misinterpreted as negative because of biased training data.

Explanation: If certain words are commonly linked to specific emotions in the training data, the model may misapply these associations in real contexts.

- Unstable Confidence When Tokens Are Removed

Example: Confidence spikes after removing noisy words like "behavior!" yet drops drastically when key words are removed.

Explanation: The model’s behavior can be unpredictable when key or irrelevant tokens are removed, leading to unreliable predictions.

- Misinterpretation of Personal and Subjective Context

Example: The model may view personal pronouns like "me" or "I" as reducing emotional intensity.

Explanation: The model fails to correctly interpret subjective expressions of emotion tied to the speaker’s perspective.

- Token-Level Focus Over Phrase-Level Meaning

Example: The model emphasizes "tomatoes" for disgust without understanding "rotten tomatoes" as a whole phrase.

Explanation: The model often treats tokens independently and overlooks the greater emotional meaning conveyed by entire phrases.

- Robustness-Accuracy Trade-off

Example: The model can give high-confidence predictions even when irrelevant tokens are removed, masking its fragility in noisy environments.

Explanation: While the model may appear confident, it is not always accurate, especially when dealing with noisy or incomplete inputs.

- Over-Distributed Relevance Across Neutral Words

Example: The model assigns similar importance to words like "the" and "astonished."

Explanation: By spreading attention too evenly, the model fails to highlight the most important emotional cues, reducing its decision-making transparency.

-- Please check this link for a thorough and complete analysis with graphs and step by step interpretations: https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group_26_y2c/blob/main/XAI_Roberta.pdf


### Recommendations for Use:

Problem Solved: Manual emotion tagging in media content (e.g., news articles, podcasts) is time-consuming and subjective.

Users deploying this emotion classification model should ensure NVIDIA GPU infrastructure (L40S or equivalent with CUDA 12.4+) supports batch processing (16/32 for train/inference) while monitoring GPU memory (keep below 85% of 46GB VRAM) and thermal limits (watch P0 power spikes near 350W). Expect ~69% accuracy and 0.68 F1-weighted performance on social media text. The model was trained using NVIDIA L40S GPUs with substantial memory (up to 46 GB). While deployment does not require this scale, using a GPU-backed instance (e.g., A10G, T4, or A100) is advised for real-time inference or large batch processing. 

Risks: Retraining or fine-tuning may be necessary when applied to new specific domains of long-form content. Language models may carry cultural or demographic biases (e.g., interpreting assertive language as anger). Interpret results with this in mind. Pretrained on Twitter so the model may underperform on medical, legal, or poetic texts. Emotion labels are mutually exclusive; mixed-emotion texts may lead to poor predictions. 	Social media-derived training data may encode demographic, gender, or cultural biases. Slight plateau in accuracy/F1 after epoch 5–8 suggests potential risk of overfitting on domain-specific val data. Output labels lack intensity/confidence scoring. It will require huma oversight for complex corpora of text as it can introduce noise.

Use Cases for Media Companies

Classify reader comments, replies, or social media mentions to segment sentiment/emotion by content type or publishing vertical (e.g., news vs entertainment).

Marketing & Brand Managers: Can use emotion insights to track campaign tone perception.

Customer Support: Can prioritize emotionally negative interactions for human escalation.

Data Analysts: Can enrich behavioral analysis pipelines with emotional tagging.

Editorial & Product Teams: May tailor tone or themes based on emotion trends over time.

Media Companies: Analyze viewer sentiment to optimize release strategies and target messaging.

### Sustainibility Considerations

GPU Power Draw (Training): 290W (NVIDIA L40S at 98% utilization), Only GPU 0 was actively involved in training.

Training Time: 25.36 minutes ≈ 0.4227 hours

CO₂ Emissions = 0.1226 kWh × 0.4kg/kWh = 0.049kg CO₂

A model deployed for 1 year at 1K daily inferences adds only ~0.3 kg CO₂ total.

Equivalent Activities:

Charging a smartphone (~12 Wh battery) ~10 times.

Running a microwave (1 kW) for 7.3 minutes.

### Recommendations for Minimizing Environmental Impact

Early Stopping: Halt training at Epoch 8, saving 5 minutes (~0.1 kWh).

Mixed Precision (FP16): Reduces GPU memory usage by 30%, lowering power draw.

Inference optimization: Train a smaller model (e.g., TinyBERT, Logistic regression) for non-critical tasks, reducing energy 4×.

OpenAI. ChatGPT. Making sure the code runs with no errors and perfoms tasks accordingly. Prompt: ‘Code optimization and intuitive variable names for all iterations', (07-03-2025).

OpenAI. ChatGPT. Reformulation, summarization and reprhasing of written aspects. Prompt: ‘XAI interpretation of explained images, error analysis, risks and limitations, use cases', (01-04-2025).