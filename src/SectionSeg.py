import pandas as pd
import string
import re

def report_split_old(txt):
    txt = txt.encode("ascii", errors="ignore").decode()
    txt = txt.lower()
    txt = txt.replace('\n', ' ')
    txt = txt.replace('\r', ' ')
    txt = txt.replace('\t', ' ')
    re1 = '(\\()'  # Any Single Character 1
    re2 = '.*?'  # Non-greedy match on filler
    re3 = '(\\))'  # Any Single Character 2

    rg = re.compile(re1 + re2 + re3, re.IGNORECASE | re.DOTALL)
    tags = ['indication: ','technique: ','comparison: ','findings: ', 'impression:']
    sections = {'indication: ': ' ','technique: ': ' ','comparison: ': ' ','findings: ': ' ', 'impression: ': ' '}
    for t in sections.keys():
        try:
            tmp = txt.split(t)[1]
            for l in tags:
                if t !=l:
                    tmp = tmp.split(l)[0]
            #sections[t]=re.sub(rg, ' ', tmp.split('legend: ')[0])
            sections[t] = re.sub(rg, ' ', tmp.split('these findings: ')[0])
        except:
            sections[t] = ' '
    print(sections.keys())
    return sections

def complex_split(critical):
    indication= []
    technique = []
    comparison = []
    findings = []
    impression = []
    report = []
    category = []
    PUI = ['pui', 'person under investigation', 'covid']
    CP =  ['chest pain']
    Cough =  ['cough']
    SOB= ['shortness of breath', 'sob', 'respiratory distress', 'chf', 'congestive, dyspnea', 'volume overload']
    Infx = ['fever', 'infection', 'sepsis']
    Neuro = ['AMS', 'headache', 'stroke', 'weakness', 'snycope', 'altered',' mental']
    Abd=  ['abdominal', 'pelvic', 'vomiting', 'distension'] 
    flg_pu = 0
    for i in range(critical.shape[0]):
        sections = extract(critical.iloc[i]['ContentText'])
        indication.append(sections['clinical indication: '])
        technique.append(sections['support devices: '])
        comparison.append(sections['comparison: '])
        findings.append(sections['findings: '])
        impression.append(sections['impression: '])
        if bool([ele for ele in PUI if(ele in sections['clinical indication: '])]):
            category.append('PUI')
            flg_pu = 1
        elif bool([ele for ele in SOB if(ele in sections['clinical indication: '])]) :
            category.append('SOB')
            flg_pu = 1
        elif bool([ele for ele in Cough if(ele in sections['clinical indication: '])]) :
            category.append('Cough')
            flg_pu = 1
        elif bool([ele for ele in Infx if(ele in sections['clinical indication: '])]) :
            category.append('Infx')
            flg_pu = 1
        elif bool([ele for ele in CP if(ele in sections['clinical indicsation: '])]) :
            category.append('CP')
            flg_pu = 1
        elif bool([ele for ele in Neuro if(ele in sections['clinical indication: '])]) :
            category.append('Neuro')
            flg_pu = 1
        elif bool([ele for ele in Abd if(ele in sections['clinical indication: '])]) :
            category.append('abd')
            flg_pu = 1
        else:
            category.append('Others')
        flg_pu = 0
    critical['indication'] = indication
    critical['support_devices'] = technique
    critical['comparison'] = comparison
    critical['indication_category'] = category
    critical['impression'] = impression
    critical['findings'] = findings
    return critical


def recieve_data(reports):
    reports = reports.fillna('N/A')
    impression = []

    for i in range(reports.shape[0]):
        sections = report_split_old(reports.iloc[i]['ContentText'])
        impression.append(sections['impression: '])
    reports['impression'] = impression
    return reports