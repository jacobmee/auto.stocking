#A股票行情数据获取演示   https://github.com/mpquant/Ashare
from  Ashare import *



df=get_price('000568.XSHG',frequency='1d',count=20)      #支持'1d'日, '1w'周, '1M'月  
print('上证指数日线行情\n',df)
