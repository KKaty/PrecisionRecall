from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from xlrd import open_workbook
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import numpy as np
import scipy as sp

filePath = raw_input("Digit the excel file path.\n")
fileName = raw_input("Digit the excel file name with its extension.\n")
wb = open_workbook(filePath+'\\'+fileName)

lowerBound, upperBound = [int(x) for x in raw_input("Digit the range of the sheets to consider separated by a whitespace : ").split()]

precisions = []
sum_precision = 0.0
sheetsConsidered = range(int(lowerBound), int(upperBound))

for sheetNumber in sheetsConsidered:

    sheet = wb.sheet_by_index(sheetNumber)
    number_of_rows = sheet.nrows
    number_of_columns = sheet.ncols
    print sheet.name
    print number_of_rows
    y_score = np.array([])
    y_true = np.array([])

    items = []
    rows = []

    columnScore = 13
    columnTrue = 14

    xRecall = np.linspace(0, 1, num=100, endpoint=True)

    for row in range(6, number_of_rows):
        score = float((sheet.cell(row,columnScore).value))
        true = int((sheet.cell(row,columnTrue).value))

        y_score = np.append(y_score, score)
        y_true = np.append(y_true, true)

    sum_precision += average_precision_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    precisionSort = precision[::-1]
    recall = sorted(recall)

    recall_copy = np.array([])
    precision_copy = np.array([])


    for i in range(len(recall)):
         if recall[i] not in recall_copy:
             recall_copy = np.append(recall_copy, recall[i])
             precision_copy = np.append(precision_copy, precisionSort[i])

    if(len(recall_copy)>2):
        f2 = interp1d(recall_copy, precision_copy, kind='cubic')
    else:
        f2 = interp1d(recall_copy, precision_copy)
    precisions.append([f2(xRecall)])


precisionsSum = [sum(x) for x in zip(*precisions)]
average_precision = sum_precision/len(sheetsConsidered)


average = [x / len(sheetsConsidered) for x in precisionsSum[0]]

averageInterp = interp1d(xRecall, average, kind='cubic')



plt.plot(xRecall, averageInterp(xRecall), 'r-')
#plt.step(recall, precision, color='b', alpha=0.2, where='post')
#plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision',fontsize=12)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision),fontsize=15)
plt.show()


print('Average precision-recall score: {0:0.2f}'.format(average_precision))