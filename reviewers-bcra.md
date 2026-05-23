Manuscript Number: BCRA-D-26-00395   

SCAFE: Compilation-Free Smart Contract Vulnerability Detection via Structural and Semantic Feature Engineering 

Dear MS.C Doitshman,    
 
Thank you for submitting your manuscript to Blockchain: Research and Applications.  

I regret to inform you that the reviewers recommend against publishing your manuscript, and I must therefore reject it. My comments, and any reviewer comments, are below.     

For alternative journals that may be more suitable for your manuscript, please refer to our Journal Finder (http://journalfinder.elsevier.com?dgcid=eman:jf-editor-reject-norev_email). 

We appreciate you submitting your manuscript to Blockchain: Research and Applications and thank you for giving us the opportunity to consider your work.  
  
Kind regards,     
Jiachi Chen   
Youth EBM Editor   
Blockchain: Research and Applications     

Editor and Reviewer comments:    
 


Thank you for submitting your manuscript to our journal. Based on the reviewers’ evaluations and the editorial assessment, we regret to inform you that we are unable to accept the manuscript for publication.

Although the reviewers recognized the practical motivation of the work, including its compilation-free design, lightweight deployment, and effort to avoid data leakage through project-based cross-validation, they also identified several substantial methodological and empirical concerns that prevent the manuscript from meeting the standard required for publication in its current form.

In particular, the reviewers raised serious concerns regarding the construction and use of the FORGE dataset, especially the mapping of “unknown” samples to negative labels and the exclusion of the “suspicious” class, which may introduce significant label noise and distort the validity of the reported performance. The evaluation protocol was also found to be problematic, as the classification threshold was selected on the held-out test fold, resulting in data leakage and limiting the real-world interpretability of F1, precision, and recall results.

The reviewers further noted that the proposed structural feature extraction relies heavily on regular expressions and a lightweight non-compiling parser, but the manuscript does not sufficiently validate the coverage, precision, or failure rate of these extracted features. This raises concerns about the reliability and reproducibility of the structural stream, particularly for modern Solidity contracts involving proxy patterns, delegatecall, interfaces, inline assembly, and newer Solidity syntax.

In addition, the reviewers found that the experimental comparison is not sufficiently up to date. The manuscript does not adequately compare or position SCAFE against recent LLM-based, neuro-symbolic, agent-based, retrieval-augmented, or formal-verification-oriented approaches to smart contract analysis. Given the rapid development of this area, the absence of such comparisons weakens the empirical and conceptual positioning of the work.

Finally, the reviewers considered several claims to be overstated, particularly those related to auditor-actionable interpretability. The use of global Random Forest feature importance does not provide contract-level explanations or guide auditors to specific vulnerable code regions. The manuscript would therefore require substantial revision to clarify the task setting, vulnerability labels, dataset semantics, feature reliability, evaluation protocol, related work, and interpretability claims.

In light of the breadth and seriousness of these concerns, we regret that we must decline the manuscript. We encourage the authors to carefully consider the reviewers’ comments when preparing a substantially revised version for future submission.
     

Reviewer 1: The submitted manuscript introduces SCAFE, a compilation-free machine learning framework designed for contract-level vulnerability detection directly from Solidity source code. The positive aspects of this work include the mitigation of compiler dependency resolution bottlenecks, the fast and resource-efficient nature of the pipeline (executing entirely on a standard CPU), and the rigorous application of project-based cross-validation splits to prevent data leakage across audited projects. However, the manuscript suffers from severe methodological problems, as follows:

 -  The authors compare SCAFE exclusively against TextCNN (a 2014 architecture), BiLSTM (1997), CodeBERT (2020), and UniXcoder (2022). They position their work as a compilation-free alternative to deep learning, but fail to acknowledge the explosion of neuro-symbolic and generative AI agents that have recently dominated this exact sub-field. The manuscript must be updated to reference and contrast against: Trace (IEEE TSE 2026), a framework that explicitly addresses the non-compilable repository problem by utilizing LLMs to synthesize missing snippets for control-flow graph construction, representing a direct methodological competitor in the compilation-free domain; SolAgent (2026) - A multi-agent framework utilizing a dual-loop refinement mechanism that integrates the Forge compiler environment and static analyzers, representing the current pinnacle of agentic vulnerability detection; LLMSA / LLMDFA (NeurIPS 2024 / 2025) - Compositional neuro-symbolic approaches explicitly
designed for compilation-free static analysis via data-flow summarization; PropertyGPT (NDSS 2025) : A framework demonstrating LLM-driven formal verification of smart contracts through retrieval-augmented property generation, showcasing the current limits of LLMs in replacing traditional static analysis.

- The manuscript explicitly states that FORGE contains 4,872 contracts categorized into three labels: "vulnerable" (1,244), "suspicious" (2,505), and "unknown" (1,123). To force this tripartite dataset into a binary classification schema, the authors discarded the "suspicious" class and mapped the "unknown" class directly to the "negative" label. Assigning a definitive negative label to an unverified, "unknown" sample is statistically invalid. The authors themselves implicitly acknowledge the danger of this failure in Section 5.2, conceding that "FORGE's unknown contracts may themselves be partially vulnerable". Training a supervised machine learning model to recognize an unknown, potentially vulnerable contract as a ground-truth "clean" contract completely distorts the mathematical decision boundary. It forces the model to artificially suppress the predicted probabilities for high-risk structural patterns simply because those patterns appeared in the contaminated "unknown"
set. Consequently, the reported ROC-AUC (0.932) and PR-AUC (0.589) metrics on the merged dataset do not reflect the model's actual vulnerability detection capabilities. Instead, they reflect the model's ability to memorize the arbitrary data collection artifacts and label noise introduced by the FORGE dataset construction. The FORGE dataset was explicitly designed as an LLM-driven framework utilizing Semantic Chunkers and MapReduce Extractors to classify CWEs ; mapping its unverified outputs to absolute negative ground truths is a fundamental misunderstanding of dataset semantics.

- The architectural foundation of the structural stream relies on regular expressions to extract function declarations, variable state modifications, and SWC vulnerability templates, alongside a lightweight, non-compiling parser (python-solidity-parser). This approach is theoretically unsound for a Turing-complete language running on the EVM. In Solidity, regular expressions are fundamentally inadequate for generating accurate call graphs or data-flow indicators due to several language-specific paradigms:
  + A regex cannot resolve which implementation of an interface is being called at runtime.
  + The use of delegatecall inside proxy patterns (which constitute a massive portion of modern upgradeable DApps) dynamically loads logic from an implementation address stored in a specific storage slot. Regex cannot traverse this boundary, meaning the call graph is silently truncated at the proxy layer.
  + Modern gas-optimized contracts rely heavily on inline assembly (assembly {... }). Regular expressions designed for high-level Solidity syntax fail completely when analyzing low-level memory operations (mload, sstore, calldatacopy). Furthermore, the python-solidity-parser utilized by the authors  is an experimental Python port of a JavaScript ANTLR parser. The literature has repeatedly documented that such experimental parsers struggle to accurately model complex Solidity 0.8.x syntax, frequently failing on custom errors, user-defined value types, and nested assembly block complexities. Constructing 306 structural features on a foundation of lossy regex and an experimental AST parser means the structural stream is highly fragile. The authors neither measure nor report the silent feature extraction failure rates of this parser, making the reproducibility and reliability of the structural stream highly suspect.

- In Section 4.4, the authors define their evaluation protocol: "F1 score, Precision, Recall: at the per-fold threshold that maximizes F1 on the held-out test fold. This is an oracle threshold that yields an upper bound on single-operating-point performance".
Tuning the classification threshold on the test set is a textbook example of data leakage. While the authors defend this practice by applying it equally to all baselines , this methodology strips the F1, Precision, and Recall metrics of all real-world validity and predictive value. In a deployed, production environment, a security practitioner cannot know the optimal threshold for unseen data because the ground truth labels are, by definition, unknown. A rigorous and scientifically valid evaluation must tune the threshold on an isolated validation set (or via nested cross-validation procedures) and apply that frozen, calibrated threshold to the test set.

- A primary claim of the manuscript is that SCAFE provides "interpretable, auditor-actionable risk factors" , positioned as a distinct advantage over opaque deep learning models. However, the interpretability mechanism provided in the paper is merely the global Gini impurity importance derived from a Random Forest (Table 10). Global feature importance indicates which features split the dataset best on average across the entire corpus; it does not explain why a specific, individual contract was flagged. Telling an auditor that a contract was flagged because Halstead volume and Trigram diversity are globally important features does not guide the auditor to the vulnerable line of code, nor does it explain the specific logical flaw. True interpretability in machine learning requires local attribution methods (such as SHAP values, LIME, or attention-weight mapping in graph networks), which are entirely absent from this work. The claim of auditor-actionability is therefore vastly
overstated.



Reviewer 2: 1.The work is generally understandable and the authors clearly communicate the motivation and overall design of SCAFE. However, several sections contain repetitive descriptions of the framework's advantages (e.g., compilation-free design, lightweight deployment, and dual-stream complementarity), which makes the presentation somewhat verbose and occasionally interrupts the logical flow.
I recommend condensing repeated statements and improving the organization of several paragraphs to make the paper more concise and easier to follow. A more streamlined presentation would significantly improve readability without affecting technical clarity.

2.This compilation-free design is practically valuable and well-motivated. However, compared with compiler-based semantic analysis tools (e.g., Slither), the proposed regex- and lightweight AST-based feature extraction approach leaves its semantic modeling capability insufficiently validated.

In particular, it remains unclear whether the extracted features can accurately and comprehensively capture real contract semantics in terms of coverage and extraction precision, or whether their effectiveness is comparable to compiler-assisted analysis methods. This raises concerns about the reliability of the feature representations used for downstream classification.

A comparative evaluation against compiler-based tools on a representative subset of contracts-particularly focusing on feature coverage and extraction consistency-would help better justify the effectiveness and limitations of the proposed compilation-free approach.

3. large language model(LLM)-based approaches for smart contract analysis have shown strong performance in recent studies, particularly in capturing richer semantic and contextual information compared to traditional feature engineering methods.
However, this paper does not include any comparison with LLM-based baselines or tools. Even a small-scale experimental comparison would be valuable to better position the proposed method, and to more clearly demonstrate its strengths and limitations in relation to modern semantic understanding approaches.
Including such a comparison would significantly strengthen the empirical evaluation and provide a more comprehensive view of the proposed method's effectiveness.

4.The manuscript seems not to clearly specify the exact vulnerability types involved or the task setting. Although the sources of the datasets are provided, the overall objective of the study and the label definitions associated with different datasets still appear to be insufficiently clarified. It would be helpful to further explain whether the proposed framework focuses on specific vulnerabilities, general vulnerability detection, or binary vulnerable/non-vulnerable classification, together with a clearer description of the label definitions used in each dataset.

5.The related work section introduces LLM-based approaches, but the discussion of recent developments such as agent-based collaboration, multi-stage reasoning, retrieval-augmented generation (RAG), and other emerging LLM-based vulnerability detection frameworks still appears somewhat incomplete. It would be helpful to include several representative works in these directions and further clarify the characteristics and differences among these approaches, in order to improve the completeness and timeliness of the related work review.


    
More information and support  

You will find information relevant for you as an author on Elsevier’s Author Hub: https://www.elsevier.com/authors 
FAQ: How can I reset a forgotten password?
https://service.elsevier.com/app/answers/detail/a_id/28452/supporthub/publishing/
For further assistance, please visit our customer service site: https://service.elsevier.com/app/home/supporthub/publishing/
Here you can search for solutions on a range of topics, find answers to frequently asked questions, and learn more about Editorial Manager via interactive tutorials. You can also talk 24/7 to our customer support team by phone and 24/7 by live chat and email


At Elsevier, we want to help all our authors to stay safe when publishing. Please be aware of fraudulent messages requesting money in return for the publication of your paper. If you are publishing open access with Elsevier, bear in mind that we will never request payment before the paper has been accepted. We have prepared some guidelines (https://www.elsevier.com/connect/authors-update/seven-top-tips-on-stopping-apc-scams ) that you may find helpful, including a short video on Identifying fake acceptance letters (https://www.youtube.com/watch?v=o5l8thD9XtE ). Please remember that you can contact Elsevier s Researcher Support team (https://service.elsevier.com/app/home/supporthub/publishing/) at any time if you have questions about your manuscript, and you can log into Editorial Manager to check the status of your manuscript (https://service.elsevier.com/app/answers/detail/a_id/29155/c/10530/supporthub/publishing/kw/status/).

__________________________________________________
In compliance with data protection regulations, you may request that we remove your personal registration details at any time.  (Use the following URL: https://www.editorialmanager.com/bcra/login.asp?a=r). Please contact the publication office if you have any questions.