library( visualFields )
#help( visualFields )
vf <- vfread('df4R_pred.csv')
# vfwrite ( vfdata, 'tmp.csv')
# vfsfa ( vfdata[1,], 'Figure01.pdf')
td <- gettd(vf)
tdp <- gettdp(td)
pd <- getpd(td)
pdp <- getpdp(pd)

vfwrite ( td, 'df4R_pred_td.csv')
vfwrite ( tdp, 'df4R_pred_tdp.csv')
vfwrite ( pd, 'df4R_pred_pd.csv')#df4R_num_pdp
vfwrite ( pdp, 'df4R_pred_pdp.csv')
