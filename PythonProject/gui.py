import tkinter as tk
import csv

gui = tk.Tk()

def enter_button():
  loan_id = Loan_ID.get()
  gender = Gender.get()
  married = Married.get()
  dependents = Dependents.get()
  education = Education.get()
  self_employed = Self_Employed.get()
  applicantincome = ApplicantIncome.get()
  coapplicantincome = CoapplicantIncome.get()
  loanamount = LoanAmount.get()
  loanamountterm = Loan_Amount_Term.get()
  credithistory = Credit_History.get()
  propertyarea = Property_Area.get()
  with open('test.csv', 'a') as f:
    w = csv.writer(f,dialect='excel-tab')
    w.writerow([loan_id,',',gender,',',married,',',dependents,',',education,',',self_employed,',',applicantincome,',',coapplicantincome,',',loanamount,',',loanamountterm,',',credithistory,',',propertyarea])

  

  
gui.geometry("800x200")
labelLoan_ID = tk.Label(gui, text = "Loan ID:")
labelLoan_ID.grid(row=0,column=0)
labelGender = tk.Label(gui, text = "Gender:")
labelGender.grid(row=0,column=2)
labelMarried = tk.Label(gui, text = "Married (Y/N):")
labelMarried.grid(row=1,column=0)
labelDependents = tk.Label(gui, text = "Dependents:")
labelDependents.grid(row=1,column=2)
labelEducation = tk.Label(gui, text = "Education (Graduate/Undergraduate):")
labelEducation.grid(row=2,column=0)
labelSelf_Employed = tk.Label(gui, text = "Self Employed (Y/N):")
labelSelf_Employed.grid(row=2,column=2)
labelApplicantIncome = tk.Label(gui, text = "Applicant Income:")
labelApplicantIncome.grid(row=3,column=0)
labelCoapplicantIncome = tk.Label(gui, text = "Coapplicant Income:")
labelCoapplicantIncome.grid(row=3,column=2)
labelLoanAmount = tk.Label(gui, text = "Loan Amount:")
labelLoanAmount.grid(row=4,column=0)
labelLoan_Amount_Term = tk.Label(gui, text = "Loan Amount Term:")
labelLoan_Amount_Term.grid(row=4,column=2)
labelCredit_History = tk.Label(gui, text = "Credit History:")
labelCredit_History.grid(row=5,column=0)
labelProperty_Area = tk.Label(gui, text = "Property Area:")
labelProperty_Area.grid(row=5,column=2)


sbmitbtn = tk.Button(gui, text = "Submit", activebackground = "green", activeforeground = "blue", comman = enter_button)
sbmitbtn.grid(row=9,column=2)

Loan_ID = tk.Entry(gui)
Loan_ID.grid(row=0,column=1)
Gender = tk.Entry(gui)
Gender.grid(row=0,column=3)
Married = tk.Entry(gui)
Married.grid(row=1,column=1)
Dependents = tk.Entry(gui)
Dependents.grid(row=1,column=3)
Education = tk.Entry(gui)
Education.grid(row=2,column=1)
Self_Employed = tk.Entry(gui)
Self_Employed.grid(row=2,column=3)
ApplicantIncome = tk.Entry(gui)
ApplicantIncome.grid(row=3,column=1)
CoapplicantIncome = tk.Entry(gui)
CoapplicantIncome.grid(row=3,column=3)
LoanAmount = tk.Entry(gui)
LoanAmount.grid(row=4,column=1)
Loan_Amount_Term = tk.Entry(gui)
Loan_Amount_Term.grid(row=4,column=3)
Credit_History = tk.Entry(gui)
Credit_History.grid(row=5,column=1)
Property_Area = tk.Entry(gui)
Property_Area.grid(row=5,column=3)



gui.mainloop()