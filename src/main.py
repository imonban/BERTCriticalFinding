import pandas as pd
from SectionSeg import complex_split
from BERT_modeltraining import modeltraining
from ClinicalBERT_training import clinicalmodeltraining

critical = pd.read_excel('Radiologyreports.xlsx') 
#'ContentText' has be list as field which will store the radiology reports
critical = critical.fillna('N/A')
critical = complex_split(critical)

modeltraining(critical)
clinicalmodeltraining(critical)