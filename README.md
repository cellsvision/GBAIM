# GBAIM

This repository contains the code for the paper **"A multi-class gastric biopsy artificial intelligence model (GBAIM) developed from whole slide histopathological images: a retrospective and prospective study"**.  

## Abstract  

Background and aim: Pathological diagnosis of biopsies is the gold standard for identifying gastric diseases, particularly gastric carcinoma, but it suffers from clinical misdiagnoses and heavy workload. Furthermore, accurately distinguishing between early-stage gastric carcinoma (EGC) and advanced-stage gastric carcinoma (AGC) during diagnosis is essential for precise treatment; however, it remains a challenge in biopsies. In this study, we aim to develop an artificial intelligence (AI) model to address these challenges.  
Methods: Data from one internal and five external medical centers, spanning 2017 to 2022, were retrospectively collected, comprising 20,711 whole-slide images (WSIs) from 17,086 patients. In 2023-2024, a prospective cohort of 3,698 WSIs from 2,965 patients was additionally enrolled. Gastric Biopsy AI Model (GBAIM) were developed to perform six-class classification on WSIs and further fine-tuned to differentiate EGC from AGC. Besides, an auxiliary diagnostic experiment was conducted with 300 WSIs and nine pathologists who were grouped as junior, intermediate and senior pathologists.   
Results: GBAIM achieved 96.1% sensitivity and 95.0% specificity in external cohorts, and 93.4% and 99.0%, respectively, in the prospective cohort, demonstrating excellent performance and strong generalization ability. The fine-tuned GBAIM-T achieved an AUC of 0.907 in EGC/AGC classification on the internal testing set, outperforming intermediate and senior pathologists, and achieved an AUC of 0.826 on the independent external testing set. GBAIM enhanced the performance of all nine pathologists in auxiliary diagnostic experiment, raised their accuracy by 1.7%-39.0%, while reducing diagnostic time by 29.4%-50.5%.   
Conclusions: GBAIM is proved to be a valuable AI tool in clinical practice. Moreover, GBAIM is, to our best knowledge, the first model capable of GC-staging in gastric biopsies.   

## Code Availability  

The code is currently being organized and will be made available before the publication of the paper. Stay tuned for updates.  
