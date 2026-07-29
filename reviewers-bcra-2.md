CC: chunming.rong@uis.no, chenjch86@mail.sysu.edu.cn

Manuscript Number: BCRA-D-26-00585   

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
 


Thank you for submitting your manuscript. Although the paper addresses a relevant problem and proposes a compilation-free framework for smart-contract vulnerability classification, the reviewers identified several fundamental concerns. These include potentially incorrect or unverifiable references, insufficient coverage of recent related work, incomplete and outdated experimental baselines, inadequate justification of the high-dimensional feature representation, and an unfair comparison with Transformer-based models that process substantially truncated inputs. The manuscript also contains multiple inconsistencies regarding feature dimensions, Solidity version coverage, metadata usage, and experimental results, which directly affect the interpretation and reproducibility of the findings.

More importantly, the dataset construction introduces a serious risk of source-label confounding, as dataset membership may be highly predictive of the vulnerability label. The merged cross-validation setting does not address this issue because samples from both datasets appear in both the training and test folds, while the poor cross-dataset results substantially weaken the claimed generalization capability. Addressing these concerns would require fundamental changes to the dataset design, evaluation protocol, baseline implementation, feature analysis, and manuscript organization, rather than a conventional revision. Therefore, the manuscript cannot be accepted for publication in its current form.
     

Reviewer 1: The paper presents SCAFE, a compiler-free binary classifier that combines 306 source-derived structural features with 73 identifier features and 50 latent semantic topics for contract-level Solidity vulnerability triage. The paper suffers from the following limitations:

 - Incorrect or fake references: [19-21], [44], [46], [47], [48], [49].
 - At minimum, the related work and baseline rationale should engage with: iAudit (ICSE 2025), which combines fine-tuning and agents and compares CodeBERT, GraphCodeBERT, CodeT5, UniXcoder, GPT-4, and CodeLlama; SAEL (ICSME 2025), an adaptive mixtureof-experts approach; PropertyGPT (NDSS 2025); the ASE 2025 access-control benchmark; TRACE (IEEE TSE 2026); and the current TOSEM review of learning-based smart-contract vulnerability detection, DOI https://doi.org/10.1145/3750042. These works do not necessarily share the same task/dataset, but they define the 2025-2026 landscape and show why CodeBERT/UniXcoder plus zero-shot 7B models are not a complete SOTA comparison.
 - The strict merged set contains 20,509 DAppSCAN clean contracts, 948 DAppSCAN vulnerable contracts, and 1,244 FORGE vulnerable contracts, while FORGE suspicious/unknown examples are excluded. Consequently, dataset membership is highly predictive of the positive label. Structural style, identifier conventions, audit-report extraction, file packaging, length, comments, and version distributions can all reveal source.
 - Merged 5-fold CV is repeatedly described as resolving the transfer gap, yet training and test folds both contain samples from both sources. True transfer is poor: the text reports DAppSCAN-to-FORGE ROC-AUC 0.492 and FORGE-to-DAppSCAN ROC-AUC 0.643/F1 0.151. These numbers undermine the generalization claim. A valid test requires a held-out source, temporal cohort, or audit-firm/developer cohort with both positives and negatives.
 - Table 5 gives DAppSCAN-trained recall of 0.100 on FORGE-vulnerable, whereas the adjacent text claims 99.9% recall and Figure 2 discusses a different vulnerable-vs-unknown evaluation. The manuscript alternates among 429, 341, 306, 292, and 203 features; Section 5.8 says 429 equals 306 + 35 + 123, which is arithmetically 464. Figure 6 includes a verified source, an Etherscan-dependent metadata feature, despite the headline source-only/API-free representation and elsewhere saying that feature is excluded. These are not cosmetic inconsistencies because they change the evaluated input and result interpretation.


Reviewer 2: This manuscript presents SCAFE, a compilation-free framework for contract-level vulnerability risk classification in smart contracts. The framework directly processes Solidity source code, combines 306 structural features with 123 semantic features, and employs an ensemble of tree-based classifiers to produce contract-level risk scores. It also uses SHAP to attribute individual predictions to specific features. However, the manuscript still has the following issues:

1. The reference entitled "SolAgent: Multi-agent Collaboration for Automated Smart Contract Vulnerability Detection" could not be located under the exact title provided. The authors are advised to verify its title, publication venue, and research scope, and to carefully check the authenticity and accuracy of all references.

2. The related-work section does not adequately cover recent developments from 2025-2026. The authors should include representative studies on smart contract vulnerability detection and multi-agent security analysis, such as:

   * Jie W, Qiu W, Yang H, et al. Agent4Vul: Multimodal LLM Agents for Smart Contract Vulnerability Detection[J]. Science China Information Sciences, 2025, 68(6): 160101.
   * Wei Z, Sun J, Sun Y, et al. Advanced Smart Contract Vulnerability Detection via LLM-Powered Multi-Agent Systems[J]. IEEE Transactions on Software Engineering, 2025.

3. The selection of experimental baselines is insufficient. Although Table 1 lists several related approaches, the experimental evaluation does not compare SCAFE with most of these tools. The authors should either include these methods in the empirical comparison or clearly explain why they could not be reproduced. In addition, the LLM baselines are limited to 7B models and therefore may not adequately represent the capabilities of current state-of-the-art models. The authors are encouraged to evaluate recent representative models from the GPT, Claude, DeepSeek, and GLM families. If such experiments are infeasible because of cost or access restrictions, the rationale for selecting the current models should be clearly explained, and the corresponding conclusions should be appropriately qualified.

4. The dual-stream feature representation is high-dimensional, but the necessity and rationality of this design are not sufficiently justified. Table 2 contains 429 features in total, which may increase the complexity of feature extraction, selection, and analysis while introducing substantial redundancy. The authors should explain why such a large feature set is necessary and analyze the independent contribution, correlation, and computational cost of each feature group. Additional ablation experiments are also needed to demonstrate that the current feature design is reasonable.

5. The supported Solidity version range is described inconsistently. The abstract states that SCAFE supports Solidity 0.4.x-0.8.x, whereas Section 5.8 reports coverage of Solidity 0.2.x-0.8.x. The authors should verify and standardize this information throughout the manuscript.

6. The proposed method appears to perform risk classification primarily through token-level and structural statistics, rather than explicitly modeling the source-code semantics, control flow, and data flow associated with the root causes of vulnerabilities. It therefore remains unclear whether the reported results reflect an understanding of vulnerability mechanisms. Moreover, if a smart contract is inherently unable to compile, it cannot be deployed on-chain, and the practical value of auditing such code requires further clarification. The authors should distinguish between contracts that are intrinsically uncompilable and contracts that cannot be compiled in the current environment because of missing dependencies or incompatible compiler versions. The practical value and intended application scenarios of compilation-free analysis should then be explained more clearly.

7. The manuscript introduces a large number of research questions and presents a considerable amount of experimental information. However, the Results section is overly lengthy, contains repeated content, and lacks a sufficiently clear logical structure, which makes the main findings difficult to identify. The authors should reorganize the relationships among the RQs, consolidate repetitive material, move secondary experiments to the appendix where appropriate, and more clearly emphasize the principal findings.

8. The comparison between SCAFE and the Transformer-based models is based on unequal input information. SCAFE processes complete Solidity source files, whereas CodeBERT and UniXcoder accept at most 512 BPE tokens and only the top 4 of their 12 Transformer layers are fine-tuned. TextCNN and BiLSTM process up to 1,024 tokens. Therefore, the models do not receive equivalent amounts of source-code information. The reported advantage of SCAFE on large contracts may primarily result from the truncation of long inputs by the Transformer models, rather than demonstrating an inherent complementarity between feature-engineered models and Transformers. The authors should consider chunking, sliding-window inference, or function-level aggregation to cover complete contracts and repeat the comparison under more equitable input conditions.



    
More information and support  

You will find information relevant for you as an author on Elsevier’s Author Hub: https://www.elsevier.com/authors 
FAQ: How can I reset a forgotten password?
https://service.elsevier.com/app/answers/detail/a_id/28452/supporthub/publishing/
For further assistance, please visit our customer service site: https://service.elsevier.com/app/home/supporthub/publishing/
Here you can search for solutions on a range of topics, find answers to frequently asked questions, and learn more about Editorial Manager via interactive tutorials. You can also talk 24/7 to our customer support team by phone and 24/7 by live chat and email


At Elsevier, we want to help all our authors to stay safe when publishing. Please be aware of fraudulent messages requesting money in return for the publication of your paper. If you are publishing open access with Elsevier, bear in mind that we will never request payment before the paper has been accepted. We have prepared some guidelines (https://www.elsevier.com/connect/authors-update/seven-top-tips-on-stopping-apc-scams ) that you may find helpful, including a short video on Identifying fake acceptance letters (https://www.youtube.com/watch?v=o5l8thD9XtE ). Please remember that you can contact Elsevier s Researcher Support team (https://service.elsevier.com/app/home/supporthub/publishing/) at any time if you have questions about your manuscript, and you can log into Editorial Manager to check the status of your manuscript (https://service.elsevier.com/app/answers/detail/a_id/29155/c/10530/supporthub/publishing/kw/status/).

__________________________________________________
In compliance with data protection regulations, you may request that we remove your personal registration details at any time.  (Use the following URL: https://www.editorialmanager.com/bcra/login.asp?a=r). Please contact the publication office if you have any questions.