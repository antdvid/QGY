import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class InflationData:
    @staticmethod
    def get_rpi_series(start_date, end_date):
        fname = 'data/20180710 - RPI Data.xls'
        df = pd.read_excel(fname)
        FORMAT = ['Month', 'RPI (Jan 1987=100)']
        data_selected = df[FORMAT]
        matrix = np.array(data_selected.values)
        matrix = matrix[matrix[:, 0] < end_date]
        matrix = matrix[matrix[:, 0] > start_date]
        return matrix

    @staticmethod
    def get_forward_inflation_rate(reference_date):
        if reference_date < pd.Timestamp(2015, 12, 1):
            fname = 'data/GLC Inflation month end data_1979 to 15.xlsx'
        else:
            fname = 'data/GLC Inflation month end data_2016 to present.xlsx'
        df = pd.read_excel(fname, sheet_name='2. fwd curve', header=3, index_col=0)
        maturity = np.array(df.columns)
        data = np.array(df.loc[reference_date])
        return np.column_stack([maturity, data])

def get_rpi_test():
    start_date = pd.Timestamp(2001, 12, 1)
    end_date = pd.Timestamp(2012, 12, 1)
    ref_date = pd.Timestamp(2011, 12, 1)
    rpi = InflationData.get_rpi_series(start_date, end_date)
    inf_ref = rpi[rpi[:, 0] == ref_date, 1]
    print(rpi[:,1])
    plt.subplot(1,2,1)
    plt.plot(rpi[:,0], rpi[:,1]/inf_ref)
    plt.subplot(1,2,2)
    plt.plot(rpi[1:,0], rpi[1:, 1]/rpi[:-1, 1] - 1)
    plt.show()

def get_fwd_test():
    start_date = pd.Timestamp(2001, 12, 31)
    rpi = InflationData.get_forward_inflation_rate(start_date)
    plt.plot(rpi[:,0], rpi[:,1])
    plt.show()

if __name__ == "__main__":
    get_rpi_test()


