"""Provides the shortened names of the weather files.

All available weather files are listed in the NAMES dict. The names
are organized as follows: Country specific abbreviation _ region specific
abbreviation _ city (+ addiditional information abbreviated).
"""

WEATHERNAMES = {
        "CH_ZH_Maur": "Switzerland_CHE_Maur",
        "CH_BS_Basel": "Basel_Fixed",
        "CH_TI_Bellinzona": "Bellinzona_Fixed",
        "CH_GR_Davos": "Davos_Fixed",
        "CH_GE_Geneva": "Geneva_Fixed",
        "CH_VD_Lausanne": "Lausanne_Fixed",
        "DNK_SD_Billund": "DNK_BILLUND_061040-IW2",
        "DNK_MJ_Horsens1": "DNK_MJ_Horsens.Bygholm.061020_TMYx.2003-2017",
        "DNK_MJ_Horsens2": "DNK_MJ_Horsens.Bygholm.061020_TMYx",
        "DNK_MJ_Isenvad1": "DNK_MJ_Isenvad.060680_TMYx.2003-2017",
        "DNK_MJ_Isenvad2": "DNK_MJ_Isenvad.060680_TMYx",
        "DNK_MJ_Karup1": "DNK_MJ_Karup-Midtjyllands.AP.060600_TMYx.2003-2017",
        "DNK_MJ_Karup2": "DNK_MJ_Karup-Midtjyllands.AP.060600_TMYx",
        "DNK_MJ_Lyngvig": "DNK_MJ_Lyngvig.Lighthouse.060590_TMYx",
        "DNK_MJ_Mejrup1": "DNK_MJ_Mejrup.060560_TMYx.2003-2017",
        "DNK_MJ_Mejrup2": "DNK_MJ_Mejrup.060560_TMYx",
        "DNK_MJ_Odum": "DNK_MJ_Odum.060720_TMYx.2003-2017",
        "DNK_MJ_Silstrup": "DNK_MJ_Silstrup.060190_TMYx",
        "ESP_CT_Barcelona": "ESP_Barcelona.081810_IWEC",
        "ESP_CT_Barcelona_ElPratAP1": "ESP_CT_Barcelona-El.Prat.AP.081810_TMYx.2003-2017",
        "ESP_CT_Barcelona_ElPratAP2": "ESP_CT_Barcelona-El.Prat.AP.081810_TMYx",
        "ESP_CT_Girona1": "ESP_CT_Girona-Costa.Brava.AP.081840_TMYx.2003-2017",
        "ESP_CT_Girona2": "ESP_CT_Girona-Costa.Brava.AP.081840_TMYx",
        "ESP_CT_Lleida1": "ESP_CT_Lleida.081710_TMYx.2003-2017",
        "ESP_CT_Lleida2": "ESP_CT_Lleida.081710_TMYx",
        "ESP_CT_Montseny": "ESP_CT_Parc.Natural.del.Montseny.081820_TMYx",
        "ESP_CT_Reus1": "ESP_CT_Reus.AP.081750_TMYx.2003-2017",
        "ESP_CT_Reus2": "ESP_CT_Reus.AP.081750_TMYx",
        "ESP_CT_Sabadell1": "ESP_CT_Sabadell.AP.081763_TMYx.2003-2017",
        "ESP_CT_Sabadell2": "ESP_CT_Sabadell.AP.081763_TMYx",
        "ESP_CT_Talar": "ESP_CT_Talar.081120_TMYx",
        "ESP_CT_Tortosa1": "ESP_CT_Tortosa-Ebre.Obs.082380_TMYx.2003-2017",
        "ESP_CT_Tortosa2": "ESP_CT_Tortosa-Ebre.Obs.082380_TMYx",
        "GRC_A_Athens": "GRC_Athens.167160_IWEC",
        "GRC_TC_Lamia1": "GRC_TC_Lamia.166750_TMYx.2003-2017",
        "GRC_TC_Lamia2": "GRC_TC_Lamia.166750_TMYx",
        "GRC_TC_LarisaAP1": "GRC_TC_Larisa.AP.166480_TMYx.2003-2017",
        "GRC_TC_LarisaAP2": "GRC_TC_Larisa.AP.166480_TMYx",
        "GRC_TC_NeaAnchialosAP1": "GRC_TC_Nea.Anchialos.AP.166650_TMYx.2003-2017",
        "GRC_TC_NeaAnchialosAP2": "GRC_TC_Nea.Anchialos.AP.166650_TMYx",
        "GRC_TC_SkiathosAP": "GRC_TC_Skiathos-Papadiamantis.Intl.AP.166653_TMYx",
        "GRC_TC_TanagraAP1": "GRC_TC_Tanagra.AP.166990_TMYx.2003-2017",
        "GRC_TC_TanagraAP2": "GRC_TC_Tanagra.AP.166990_TMYx",
        "GRC_TC_Trikala": "GRC_TC_Trikala.166450_TMYx",
        "USA_IL_Centralia_MuniAP": "USA_IL_Centralia.Muni.AP.744657_TMYx",
        "USA_IL_Chicago_Calumet": "USA_IL_Chicago-CGS.Calumet.725344_TMYx.2004-2018",
        "USA_IL_Chicago_ExecAP": "USA_IL_Chigago.Exec.AP.725339_TMYx.2004-2018",
        "USA_IL_Chicago_ExecAP2": "USA_IL_Chigago.Exec.AP.725339_TMYx",
        "USA_IL_Chicago_MidwayAP": "USA_IL_Chicago.Midway.Intl.AP.725340_TMY3",
        "USA_IL_Chicago_OHareAP": "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3",
        "USA_IL_Chicago_OHareAP2": "USA_IL_Chicago.OHare.Intl.AP.725300_TMYx.2004-2018",
        "USA_IL_Chicago_OHareAP3": "USA_IL_Chicago.OHare.Intl.AP.725300_TMY3",
        "USA_IL_DecaturAP": "USA_IL_Decatur.AP.725316_TMYx",
        "USA_IL_FreeportAP": "USA_IL_Freeport-Albertus.AP.722082_TMYx.2004-2018",
        "USA_NY_HudsonRiver": "USA_NY_Hudson.River.Reserve.997991_TMYx",
        "USA_NY_LongIsland_MacArthurAP1": "USA_NY_Islip-Long.Island.MacArthur.AP.725050_TMY3",
        "USA_NY_LongIsland_MacArthurAP2": "USA_NY_Islip-Long.Island.MacArthur.AP.725050_TMYx.2004-2018",
        "USA_NY_LongIsland_MacArthurAP3": "USA_NY_Islip-Long.Island.MacArthur.AP.725050_TMYx",
        "USA_NY_NewYork_KennedyAP1": "USA_NY_New.York-Kennedy.Intl.AP.744860_TMYx.2004-2018",
        "USA_NY_NewYork_KennedyAP2": "USA_NY_New.York-Kennedy.Intl.AP.744860_TMY3",
        "USA_NY_NewYork_KennedyAP3": "USA_NY_New.York-Kennedy.Intl.AP.744860_TMYx",
        "USA_NY_NewYork_LaGuardiaAP": "USA_NY_New.York-LaGuardia.AP.725030_TMYx",
        "USA_NY_Newburgh_StewartAP1": "USA_NY_Newburgh-Stewart.Intl.AP.725038_TMY3",
        "USA_NY_Newburgh_StewartAP2": "USA_NY_Newburgh-Stewart.Intl.AP.725038_TMYx.2004-2018",
        "USA_NY_Newburgh_StewartAP3": "USA_NY_Newburgh-Stewart.Intl.AP.725038_TMYx",
    }
