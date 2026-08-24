@IF NOT EXIST %1.DAT GOTO NO
@IF EXIST %1.PL4 DEL %1.PL4
@IF EXIST %1.LIS DEL %1.LIS
del *.err
set atpdir=d:\atp32\
tpbig.exe both %1.dat %1. -R
@REM ND
@IF EXIST %1.PL4 GTPPL32 %1
@GOTO END
:NO
@ECHO EL ARCHIVO %1.DAT NO EXISTE
:END
