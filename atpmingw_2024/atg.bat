@ECHO OFF
@if exist %TMP%\fort* del %TMP%\fort*
@IF NOT EXIST %1.DAT GOTO NO
@IF EXIST %1.PL4 DEL /k %1.PL4
@IF EXIST %1.LIS DEL /k %1.LIS
@IF EXIST %1.LIS DEL /k %1.DBG
@IF EXIST *.ERR DEL /k *.ERR
REM set atpdir=d:\atpnew\
tpbig.exe disk %1.dat %1. -R
@REM ND
@set gtpplot=f:\util\
@IF EXIST %1 GTPPL32 @@%1
@IF EXIST %1 GOTO END
@IF EXIST %1.PL4 GTPPL32 %1
@set gtpplot=
@GOTO END
:NO
@ECHO THE FILE %1.DAT DOESN'T EXISTS
:END
