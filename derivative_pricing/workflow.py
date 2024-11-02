from data_cleaning import data_cleaning
from implied_vol import BS_implied_vol
from fit_bs import fit_BS
from local_vol import local_vol_surface
#--------------------------------------------
import pandas as pd
from IPython.display import display
from matplotlib.dates import date2num, num2date
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np



option_data_spx = pd.read_csv('../Public/Data/Option/spx_option_0901.csv')
option_data_nky = pd.read_csv('../Public/Data/Option/nky_option_0901.csv')
option_data_hsi = pd.read_csv('../Public/Data/Option/hsi_option_0901.csv')

option_data_spx = data_cleaning(option_data_spx).format_data(index = 'SPX')
option_data_nky = data_cleaning(option_data_nky).format_data(index = 'NKY')
option_data_hsi = data_cleaning(option_data_hsi).format_data(index = 'HSI')

option_data_spx = data_cleaning(option_data_spx).check_iv_number(drop_type='volume', drop_threshold=10)
option_data_nky = data_cleaning(option_data_nky).check_iv_number(drop_type='volume', drop_threshold=2)
option_data_hsi = data_cleaning(option_data_hsi).check_iv_number()

spx_data = data_cleaning(option_data_spx).get_hist('SPX', '2021-09-01', '2023-11-13')
nky_data = data_cleaning(option_data_nky).get_hist('NKY', '2021-09-01', '2023-11-13')
hsi_data = data_cleaning(option_data_hsi).get_hist('HSI', '2021-09-01', '2023-11-13')

option_price_spx = data_cleaning(option_data_spx).extract_option_price(px_type='mid')
option_price_nky = data_cleaning(option_data_nky).extract_option_price(px_type='mid')
option_price_hsi = data_cleaning(option_data_hsi).extract_option_price(px_type='mid')



option_data = pd.concat([option_data_spx, option_data_nky, option_data_hsi], axis=0)
option_price = pd.concat([option_price_spx, option_price_nky, option_price_hsi], axis=0)

implied_params = BS_implied_vol(option_price).parity_implied_params(option_data, plot_parity=False)
implied_vol_spx = BS_implied_vol(option_price[option_price['Index'] == 'SPX']).get_iv(option_data[option_data['Index'] == 'SPX'], implied_params[implied_params['Index'] == 'SPX'], plot_iv_scatter=False)
implied_vol_nky = BS_implied_vol(option_price[option_price['Index'] == 'NKY']).get_iv(option_data[option_data['Index'] == 'NKY'], implied_params[implied_params['Index'] == 'NKY'], plot_iv_scatter=False)
implied_vol_hsi = BS_implied_vol(option_price[option_price['Index'] == 'HSI']).get_iv(option_data[option_data['Index'] == 'HSI'], implied_params[implied_params['Index'] == 'HSI'], plot_iv_scatter=False)



fwd_moneyness_spx = fit_BS(implied_vol_spx, implied_params[implied_params['Index'] == 'SPX']).get_fwd_mny()
fwd_moneyness_nky = fit_BS(implied_vol_nky, implied_params[implied_params['Index'] == 'NKY']).get_fwd_mny()
fwd_moneyness_hsi = fit_BS(implied_vol_hsi, implied_params[implied_params['Index'] == 'HSI']).get_fwd_mny()

bs_iv_curve_params_spx = fit_BS(implied_vol_spx, implied_params[implied_params['Index'] == 'SPX']).fit_BS_curve(fwd_moneyness_spx, plot_curve=False)
bs_iv_curve_params_nky = fit_BS(implied_vol_nky, implied_params[implied_params['Index'] == 'NKY']).fit_BS_curve(fwd_moneyness_nky, plot_curve=False, method='tanh')
bs_iv_curve_params_hsi = fit_BS(implied_vol_hsi, implied_params[implied_params['Index'] == 'HSI']).fit_BS_curve(fwd_moneyness_hsi, plot_curve=False)

step = 50
bs_iv_surface_spx = fit_BS(implied_vol_spx, implied_params[implied_params['Index'] == 'SPX']).fit_surface(bs_iv_curve_params_spx, plot_surface=False, step=step, type='wireframe')
bs_iv_surface_nky = fit_BS(implied_vol_nky, implied_params[implied_params['Index'] == 'NKY']).fit_surface(bs_iv_curve_params_nky, plot_surface=False, step=step)
bs_iv_surface_hsi = fit_BS(implied_vol_hsi, implied_params[implied_params['Index'] == 'HSI']).fit_surface(bs_iv_curve_params_hsi, plot_surface=False, step=step)


# for each point on the surface, calculate the local vol
local_vol_surface_spx = local_vol_surface(bs_iv_surface_spx)
local_vol_surface_nky = local_vol_surface(bs_iv_surface_nky)
local_vol_surface_hsi = local_vol_surface(bs_iv_surface_hsi)


