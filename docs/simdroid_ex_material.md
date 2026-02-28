# 材料常数

\
//  一般单元材料

//  “IsotropicElastic”/“IsotropicPlasticJC”/"RateDependentPlastic "/"HyperElasticRubber  "/"IsotropicPlasticCPRate"/"HyperFoam"/"HyperElasticRubberCurve"/" ViscoElasticFoam"


"MaterialConstants": {                                // 需按照材料的不同类型填写不同的关键字

// MaterialType == Empty【LAW0】

```javascript
 "ElasticModulus": ... ,                    // double, 弹性模量  (0, e38)【E】
 "PoissonRatio": ... ,                        // double, 泊松比   (-1, 0.5]【ν】
```


```
        // MaterialType == IsotropicElastic【LAW1】
```

```javascript
"ElasticModulus": ... ,                          // double, 弹性模量 (0, e38)【E】
"PoissonRatio": ... ,                              // double, 泊松比  (-1, 0.5]【v】
"ElasticModulusT": ... ,                       // string, 弹性模量随温度曲线
"PoissonRatioT": ... ,                            // string,  泊松比随温度曲线
```


// MaterialType == IsotropicPlasticJC【LAW2】

```javascript
"ElasticModulus": ... ,                     // double, 弹性模量  (0, e38)【E】
"PoissonRatio": ... ,                        // double, 泊松比   (-1, 0.5]【ν】
"YieldStress": ... ,                           // double, 屈服常数  （-e38, +e38）A【a】
"HardeningCoefB": ... ,                  // double, 硬化系数   （-e38, +e38）B【b】
"HardeningExpN": ... ,                   // double, 硬化幂指数  （-e38, +e38）n【n】
"HardeningMode": ... ,                  // double, 硬化模式，0各向同性【0】/1运动硬化【1】/alpha权重系数∈（0,1）【C_hard】
"RateCoef": ... ,                              // double, 应变率相关系数 （-e38, +e38）【c】
"TemperatureExp": ... ,                  // double, 温度相关幂指数  （-e38, +e38）【m】
"MeltTemperature": ... ,                // double, 融化温度  （-273, +e38）【T_melt】
"EnvTemperature": ... ,                  // double, 环境温度  （-273, +e38）【Tr】
"RefStrainRate": ... ,                      // double, 参考应变率   [0, +e38）,默认1.0【ε0_dot】
"SpecificHeat": ... ,                       // double, 比热容   （0, +e38）【ρCp】
```


//MaterialType == RateDependentPlastic【LAW36】

```javascript
"ElasticModulus": ... ,                          // double, 弹性模量  (0, e38)【E】
"PoissonRatio": ... ,                             // double, 泊松比   (-1, 0.5]【v】
"HardeningMode":...,                          // double, 0, 1, alpha(0,1)【C_hard】
"FailurePlasticStrain": ... ,                    // double, 失效塑性应变   [0, e38)【εp_max】
"FailBeginTensileStrain": ... ,               // double, 应力刚开始降低时的拉伸应变【εt】
"FailEndTensileStrain": ... ,                  // double, 应力降低为0时的拉伸应变【εm】
"StrainRateType": ... ,                         //  string, "PlastStrain【1】"/"TotalStrain"(默认)【0】//【VP】
"ElemDelTensileStrain": ... ,                // double, 单元删除时的工程应变 【εf】
"StrainAndStrainRateYieldCurve": ..., //string[], 函数,1个自变量1个因变量
"StrainRate":...                                 //double [],应变率
```


//MaterialType == HyperElasticRubber【LAW42】

```javascript
"PoissonRatio": ... ,                           // double, 泊松比   (-1, 0.5]【v】
"Mu": ... ,                                          // double[5]【μ_i】 !! 注意与abaqus的定义是不同的，势函数为 mu/alpha
"Alpha": ... ,                                      // double[5]【α_i】 
"m": ... ,                                            // int, 粘性Maxwell模型的项数【M】
"Stiff": ... ,                                      // double[m], Maxwell模型弹簧刚度系数【Gi】
"RelaxTime": ... ,                               //double[m]，Maxwell模型松弛时间【τi】
```


//MaterialType == OgdenRubber【LAW82】

```json
"MaterialConstants": {
    // 直接定义的方式
    "PoissonRatio": ... ,     // double, 泊松比   (-1, 0.5]【v】
    "Mu": ... ,               // double[]【μ_i】 !! 注意与abaqus的定义是相同的，势函数为2*mu/alpha^2
    "Alpha": ... ,            // double[]【α_i】 
    "D": ... ,                //double[]【D_i】
    // 实验曲线的方式
    "TestCurve-Uniaxial": ["curve-u"],
    "TestCurve-Biaxial": ["curve-b"],
    "TestCurve-Planar": ["curve-p"],
    "TestCurve-Volumetric": ["curve-v"],
    "CurveFit_n": 1                   // int 曲线拟合的阶数
    // Mullins效应，可实现部分参数定义，部分实验曲线定义
    "MullinsEffect":{
        //参数定义
        "r": 2.0,
        "m": 0.0,
        "beta": 0.1
        //实验曲线定义
        "TestCurve-Uniaxial": ["Uniaxial-1","Uniaxial-2"],
        "TestCurve-Biaxial": ["Biaxial-1","Biaxial-2"]
    }
}
```


// MaterialType == IsotropicPlasticCPRate【LAW44】

```javascript
"MaterialConstants": {
    "ElasticModulus": ... ,                        // double, 弹性模量  (0, e38)【E】
    "PoissonRatio": ... ,                           // double, 泊松比  (-1, 0.5]【v】
    "YieldStress": ... ,                              // double, 屈服应力  (0, e38)【a】
    "HardeningCoefB": ... ,                     // double, 硬化参数b【b】
    "HardeningExpN": ... ,                      // double, 硬化幂指数n【n】
    "HardeningMode": ... ,                    // double, 硬化模式，0各向同性硬化，1运动硬化，alpha（0,1）是两者线性组合系数【C_hard】
    "MaxStress": ... ,                               // double, 最大塑性应力【σ_max0】
    "RateCoefC": ... ,                             //double, “CP”应变率系数C【c】
    "RateCoefP": ... ,                             //double, “CP”应变率系数P【p】
    "StrainRateScaleMaxStress": ... ,     //bool, true(默认)【1】应变率对最大塑性应力做缩放，false不做缩放【2】//【ICC】
    "FailurePlasticStrain": ... ,                    // double, 失效塑性应变   [0, e38)【εp_max】
    "FailBeginTensileStrain": ... ,               // double, 应力刚开始降低时的拉伸应变【εt1】
    "FailEndTensileStrain": ... ,                  // double, 应力降低为0时的拉伸应变【εt2】
    "StrainRateType": ... ,                      //string , "PlastStrain【1】"/"TotalStrain"(默认)【2】/"DeviatoricStrain"【3】//【VP】
}
```


// MaterialType == HyperFoam【LAW62】

```json
"MaterialConstants": {
    // 直接定义的方式
    "PoissonRatio": ... ,    // double, 泊松比   (-1, 0.5]【ν】
    "Mu": ... ,              // double[N],     (0, e38)【μ_i】 !! 注意与abaqus的定义是相同的，势函数为2*mu/alpha^2
    "Alpha": ... ,           // double[N],   （-e38, +e38）【α_i】
    "ViscousForm":...        // str, "Dev"【0】或“DevVol”【1】//【Flag_Visc】
    "MaxVisco":..,           //double
    // 实验曲线的方式
    "TestCurve-Uniaxial": ["curve-u"],
    "TestCurve-Biaxial": ["curve-b"],
    "TestCurve-Planar": ["curve-p"],
    "TestCurve-Volumetric": ["curve-v"],
    "TestCurve-SimpleShear-S": ["curve-S"],
    "TestCurve-SimpleShear-T": ["curve-T"],
    "CurveFit_n": 1,      // int 曲线拟合的阶数
    "CurveFit_Nu": 0.0    // double 泊松比
}
```


          //MaterialType == HyperElasticRubberCurve【LAW69】

```javascript
"Law": ... ,                                          // Ogden, Mooney-Rivilin
"PoissonRatio": ... ,                             // double, 泊松比   (-1, 0.5]
"EngineeringStrainStressCurve": ... , // string, 函数， 1个自变量1个因变量
```


// MaterialType == ViscoElasticFoamAbq【LAW70】// 对标ABAQUS的low density foam

```javascript
"ElasticModulus": ... ,                           // double, 弹性模量  (0, e38)【E0】
"PoissonRatio": ... ,                             // double, 泊松比   (-1, 0.5]【ν】
"MaxElasticModulus":..                      //double,【Emax】,默认0
"PoissonRatioMax": ..                        //double,【νmax】，默认1            // 应该有错误，描述与手册不一致，泊松比不应该超过0.5，改为"RefStrain"
"RefStrain": …                                    // double 【emax】，默认1.0
"CompressFlag":...                            //bool , 【Itens】默认false
"LoadingCurves":..                          //string[], 加载曲线名称
"LoadingStrainRates":..                          //double[], 加载曲线应变率
"UnloadingCurves":..                          //string[], 卸载曲线名称
"UnloadingStrainRates":..                          //double[], 卸载曲线应变率
“UnloadResponseFlag”:                        //int   
"UnloadingShape": ... ,                      //  double, 卸载函数中的幂指数【Shape】
“UnloadingHysteretic”: ... ,                //  double, 卸载函数中的系数 【Hys】
```


// MaterialType == ViscoElasticFoamDyna【LAW90】    // 对标LSDYNA的low density foam

```javascript
"ElasticModulus": ... ,                          // double, 弹性模量  (0, e38)【E0】
"PoissonRatio": ... ,                             // double, 泊松比   (-1, 0.5]【ν】
"LoadingCurves": ... ,                           // string[], 曲线名称， 1个自变量1个因变量【fct_IDL】
"StrainRates":..,                                    //double[],应变率 
"UnloadingShape": ... ,                      //  double, 卸载函数中的幂指数【Shape】
"UnloadingHysteretic": ... ,                //  double, 卸载函数中的系数 【Hys】
```


// MaterialType == HyperElasticYeoh【LAW94】——暂未开发

```javascript
"DeviatoricConsts": ... ,                    // double[3], [C10, C20, C30]
"VolumetricConsts": ... ,                   // double[3], [D1, D2, D3]
```


//MaterialType == CohesiveBilinearMixedmode【LAW117】

```json
"MaterialType": CohesiveBilinearMixedmode          //MaterialType == CohesiveBilinearMixedmode【LAW117】    
    "ElasticModulus": ...        // double[2], 弹性模量（0, e38）【EN, ET】
    "DensityFlag": ...           // string, Area【1】"/"Volume【2】//【Imass】
    "IntDel": ...                // int, 4【4】//【Idel】
    "MixModeLaw": ...            // string, "PowerLaw"【1】/"B-K"【2】//【Irupt】
    "T0n": ...                   // double, peak traction in normal direction//【TN】
    "T0t": ...                   // double, peak traction in tangential direction//【TT】
    "GCn": ...                   // double, energy release rate for pure normal direction//【GIC】
    "GCt": ...                   // double, energy release rate for pure shear direction// 【GIIC】
    "EXP": ...                   // double, exponent for "PowerLaw" and "B-K"//【EXP_G】【EXP_BK】
    "Gamma": ...                 // double, default 1.0, used in "B-K" law// 【Gamma】
```


//MaterialType == Cohesive【LAW118】

```json
"MaterialType": Cohesive         //MaterialType == Cohesive【LAW118】
    "ElasticModulus": ...        // double[3], 弹性模量（0, e38），一个法向，两个切向
    "DensityFlag": ...           // string, Area【1】"/"Volume【2】//【Imass】
    "IntDel": ...                // int, 4【4】//【Idel】
    "DamageCriterion": ...       // string, "QUADS"或者“MAXS”
    "DamageCriterioncof": ...    // double[3],初始损伤准则参数
    "EvolutionType": ...         //string, “DISP”或者“ENERGY” 演化类型
    "Evolutioncof": ...          // double[1], 演化参数，类型为DISP时表示最大失效位移，ENERGY表示失效时的总能量Gc
    "NonEvolution":...           // bool,是否为非线性损伤演化
    "Alpha":...                  // double, 非线性损伤演化时的非尺度材料参数
```


//MaterialType == CohesiveTrilinearMixedmode【LAW116】

```javascript
"ElasticModulus": ...                        // double[2], 弹性模量（0, e38）【EI, EII】
"DensityFlag": ...                             // string, 面积密度or体积密度，"Area【1】"/"Volume                                                                                 【2】（默认值）"//【Imass】
"Thickness": ...                                // double, 参考厚度【Thick】
"IntDel": ...                                      //int, 单元删除需达到的失效积分点数量【Idel】
"DamageInitCrit": ...                       // string, 失效起始准则，"QuadStress【1】"/"MaxStress                                                                               【2】"//【Icrit】
"FG": ...                                           // double[2], [FGn, FGs]，屈服阶段形状系数，当FGn=FGs=0时可对标Abq//【fGI, fGII】
"DamageEvolCrit": ...                     // string,失效演化准则， "Disp【2】"/“Energy【1】”//【Ifail_I】
"rEG0": ...                                       //doulbe[2], [rEG0n, rEG0s], 失效演化率相关能量方程参数，见参数说明//【GCI_ini、GCII_ini】
"rEG1": ...                                       //doulbe[2], [rEG1n, rEG1s], 失效演化率相关能量方程参数，见参数说明//【GCI_inf、GCII_inf】
"rEdotEG": ...                                 // double[2], [rEdotEGn, rEdotEGs], 失效演化率相关能量方程参数，见参数说明//【˙εGI、˙εGII】
"rSRateOrder": ...                          // int, 1【1】/2【2】，失效演化率相关应力方程指数，见参数说明//【Iorder_i】
"rS0": ...                                        // double[2], [rS0n, rS0s], 失效演化率相关应力方程参数，见参数说明//【σA_I、σA_II】
"rS1": ...                                        // double[2], [rS1n, rS1s], 失效演化率相关应力方程参数，见参数说明//【σB_I、σB_II】
```


//MaterialType ==Dprag【LAW102】

```javascript
"ElasticModulus": ...                           //弹性模量【E】
"PoissonRatio": ... ,                        // double, 泊松比   (-1, 0.5]【ν】
"FormulationFlag":...                    //string, "Circumscribed"[1],"Middle"[2],"Inscribed"[3]
"Cohesion":...                                //double, 【c】
"InterFricAngle":...                           //double, 【pha】
"YieldLimit":...                                  //double, 【Amax】
"MinPressure":...                           //double, 【Pmin】
"Eos":..                            // string, 状态方程名字
```


//MaterialType ==HydJCook【LAW4】

```json
 "MaterialType":HydJCook,               //MaterialType ==HydJCook【LAW4】
 "MaterialConstants": {
    "RefDensity":..                    //double, pho0
    "ElasticModulus": ...              //弹性模量【E】
    "PoissonRatio": ... ,              // double, 泊松比   (-1, 0.5]【ν】
    "YieldStress":...,                 //double ,【a】
    "PlasHarden":...,                  //double,【b】
    "PlasHardenExp":...,               //double,【n】
    "FailureStrain":...                //double,【epsmax 】
    "MaxStress":...                    //double,【sigmax】
    "PressureCutoff":...               //double, 【Pmin】 
    "StrainRate": ...                  //double，【c】
    "RefStrainRate":...                //double,【EPS_DOT_0】
    "TempExp":...                      //double,【m】
    "MeltTemp":...                     //double,【Tmelt】
    "MaxTemp":...                       //double, Tmax
    "SpeciHeat":...                     //double,【RHOCP】
    "Eos":..                            // string, 状态方程名字
    "Fail":..                           //string, 失效模型名字
}
```


//-形状记忆合金【LAW71】

```json
"MaterialType":"SuperElasticSMA",      //MaterialType=SuperElasticSMA
"MaterialConstants": {
    "ElasticModulus":...,        //double, 弹性模量【E】
    "PoissonRatio": ... ,        // double, 泊松比 (-1, 0.5]【ν】    
    "StressSAS":...,             //double, 【Sigma_SAS】
    "StressFAS":...,             //double, 【Sigma_FAS】
    "StressSSA":...,             //double, 【Sigma_SSA】
    "StressFSA":...,             //double, 【Sigma_FSA】
    "Alpha":...,                 //double, 【alpha】
    "TransformStrain":...,       //double, 【EpsL】
    "StressTempRate":...,        //double[2], 加载和卸载阶段的应力-温度率【CAS,  CSA】
    "ReferenceTemp":...,         //double[4], 相变阶段的参考温度【TS_AS, TF_AS, TS_SA, TF_SA】
}
```


// - 正交异性弹性+Hill塑性【LAW93】

```json
"MaterialType": "OrthotropicElasticHillPlastic", // MaterialType ==OrthotropicElasticHillPlastic
"MaterialConstants": {
    "ConstantsType": ...,         // string, "EngineeringConstants"/"DirectConstants"
    // 工程常数的方式
    "ElasticModulus": ... ,      // double[3], 弹性模量（0, e38）【E11, E22, E33】
    "PoissonRatio": ... ,        // double[3], 泊松比 (-1, 0.5]  【ν12, ν13, ν23】
    "ShearModulus": ... ,        // double[3], 剪切模量 (0, e38)【G12, G13, G23】
    // 直接定义的方式
    "Constants": ...              // double[9], [D1111,D1122,D2222,D1133,D2233,D3333,D1212,D2323,D1313]
    // 塑性部分的参数
    "StrainRateType": ... ,      //  string , "PlastStrain【1】"/"TotalStrain"(默认) 【2】/"DeviatoricStrain"【3】//【VP】
    "YieldCurve": ... ,          // string[], 函数, 1个自变量1个因变量【fct_ID】
    "StrainRate":..,             //double[],应变率 
     "YieldStressRatio": ... ,   //  double[6]，硬化参数【R11，R22，R33，R12，R13，R23】
 }
```


// - 正交异性线弹性【LAW12】——暂未开发

```json
"MaterialType": "OrthoElastic", // MaterialType ==OrthoElastic
"MaterialConstants": {
    "ConstantsType": ...,         // string, "EngineeringConstants"/"DirectConstants"
    // 工程常数的方式
    "ElasticModulus":...          //double[3], 弹性模量[E1,E2,E3]
    "PoissonRatio":...            //double[3], 泊松比 [ν12，ν23，ν31]
    "ShearModulus": ... ,         //double[3], 剪切模量 [G12, G13, G23]
    // 直接定义的方式
    "Constants": ...              // double[9], [D1111,D1122,D2222,D1133,D2233,D3333,D1212,D2323,D1313]
}
```


//-OrthotropicElastic, 隐式移植算法，用于Tet4M单元

```json
"MaterialType": "OrthotropicElastic", // MaterialType ==OrthotropicElastic
"MaterialConstants": {
    "ConstantsType": ...,         // string, "EngineeringConstants"/"DirectConstants"
    // 工程常数的方式
    "ElasticModulus": ... ,      // double[3], 弹性模量（0, e38）【E11, E22, E33】
    "PoissonRatio": ... ,        // double[3], 泊松比 (-1, 0.5]  【ν12, ν13, ν23】
    "ShearModulus": ... ,        // double[3], 剪切模量 (0, e38)【G12, G13, G23】
    // 直接定义的方式
    "Constants": ...              // double[9], [D1111,D1122,D2222,D1133,D2233,D3333,D1212,D2323,D1313]
}
```


// - MooneyRivlin, 隐式移植的算法

```json
"Material":{
    "name1": {
        "Density": 1.2e-9,
        "MaterialType": "MooneyRivlin",
        "MaterialConstants": {
        // 直接定义的方式
            "C10": 0.181,                      // double
            "C01": 0.045,
            "D1": 0.01,
        // 实验曲线的方式
            "TestCurve-Uniaxial": ["curve-u1","curve-u2"],    // vector<string>
            "TestCurve-Biaxial": ["curve-b"],
            "TestCurve-Planar": ["curve-p"],
            "TestCurve-Volumetric": ["curve-v"],
        // Mullins效应
            "MullinsEffect":{
                "r": 2.0,
                "m": 0.0,
                "beta": 0.1
            }
        }
    }
}
```


// - Ogden2, 隐式移植的算法

```json
"MaterialType": "Ogden2",
"MaterialConstants": {
    // 直接定义的方式
    "Mu": ... ,           // vector<double>，定义的是Mu 而不是Mu^h, Mu^h = Alpha*Mu/2, Mu^h是Abaqus的定义
    "Alpha": ... ,        // vector<double>，与Mu等长
    "PoissonRatio": 0.49, // double
    // 实验曲线的方式
    "TestCurve-Uniaxial": ["curve-u"],
    "TestCurve-Biaxial": ["curve-b"],
    "TestCurve-Planar": ["curve-p"],
    "TestCurve-Volumetric": ["curve-v"],
    "CurveFit_n": 1                   // int 曲线拟合的阶数
    // Mullins效应 同 MooneyRivlin
}
```


// - Polynomial 【LAW100】

```json
"MaterialType": "Polynomial",
"MaterialConstants": {
    // 直接定义的方式
    "Order": 1,     // int 1到6
    "Const": ...    // vector<double>  对应Order=1:6 长度分别为 3/7/12/18/25/33
     // Order=1 [C10,C01,D1]
     // Order=2 [C10,C01,C20,C11,C02,D1,D2]
     // Order=3 [C10,C01,C20,C11,C02,C30,C21,C12,C03,D1,D2,D3]
     // Order=4 [C10,C01,C20,C11,C02,C30,C21,C12,C03,C40,C31,C22,C13,C04,D1,D2,D3,D4]
     // Order=5 [C10,C01,C20,C11,C02,C30,C21,C12,C03,C40,C31,C22,C13,C04,
     //          C50,C41,C32,C23,C14,C05,D1,D2,D3,D4,D5]
     // Order=6 [C10,C01,C20,C11,C02,C30,C21,C12,C03,C40,C31,C22,C13,C04,
     //          C50,C41,C32,C23,C14,C05,C60,C51,C42,C33,C24,C15,C06,D1,D2,D3,D4,D5,D6]
     
    // 实验曲线的方式 同 Ogden2
    // Mullins效应 同 Ogden2
}
```


// - ReducedPolynomial  【LAW100】

```json
"MaterialType": "ReducedPolynomial",
"Moduli":"LongTerm",//"Longterm"/"Instantaneous" default:Longterm
"PoissonRatio":0.3 // double
"MaterialConstants": {
    // 直接定义的方式
    "Order": 1,     // int 1到6
    "N_net":2,        //int
    "Const": ...    // vector<double>  对应Order=1:6 长度分别为 2*Order
     // Order=1 [C10,D1]
     // Order=2 [C10,C20,D1,D2]
     // Order=3 [C10,C20,C30,D1,D2,D3]
     // Order=4 [C10,C20,C30,C40,D1,D2,D3,D4]
     // Order=5 [C10,C20,C30,C40,C50,D1,D2,D3,D4,D5]
     // Order=6 [C10,C20,C30,C40,C50,C60,D1,D2,D3,D4,D5,D6]
     
    // 实验曲线的方式 
    "TestCurve-Uniaxial": ["curve-u"],
    "TestCurve-Biaxial": ["curve-b"],
    "TestCurve-Planar": ["curve-p"],
    "TestCurve-Volumetric": ["curve-v"],
    // Mullins效应 同 Ogden2
    
    // 如果 N_net > 0，则生成此字段
    "ViscoNetworks": {
        "network1": {
            "FlagVisc": 3, // 原始 Flag 方便追溯
            "Model": "PowerLaw", // 可读性名称
            "StiffnessWeight": 0.5,
            "Parameters": { "A3": 0.0031623, "n3": 1.0, "M3": -0.5 }
        },
        "network2": {
            "FlagVisc": 2,
            "Model": "HyperbolicSine",
            "StiffnessWeight": 0.3,
            "Parameters": { "A2": 2.5e-05, "B": 0.15, "n2": 2.0 }
        }
        // ... 更多网络
}
```

| **Flag_visc** | **模型名称** | **JSON Model** | **参数键值对** |
|----|----|----|----|
| **1** | Bergstrom-Boyce | `"BergstromBoyce"` | `A1`, `C`, `M`, `Xi`, `Tau_ref` |
| **2** | Hyperbolic Sine | `"HyperbolicSine"` | `A2`, `B`, `n2` |
| **3** | Power Law | `"PowerLaw"` | `A3`, `n3`, `M3` |


// - HyperFoam2, 隐式移植的算法

```json
"MaterialType": "HyperFoam2",
"MaterialConstants": {
    // 直接定义的方式
    "Mu": ... ,           // vector<double>，定义的是Mu 而不是Mu^h, Mu^h = Alpha*Mu/2, Mu^h是Abaqus的定义
    "Alpha": ... ,        // vector<double>，与Mu等长
    "Nu": ... ,           // vector<double>，与Mu等长
    // 实验曲线的方式
    "TestCurve-Uniaxial": ["curve-u"],
    "TestCurve-Biaxial": ["curve-b"],
    "TestCurve-Planar": ["curve-p"],
    "TestCurve-Volumetric": ["curve-v"],
    "TestCurve-SimpleShear-S": ["curve-S"],
    "TestCurve-SimpleShear-T": ["curve-T"],
    "CurveFit_n": 1,      // int 曲线拟合的阶数
    "CurveFit_Nu": 0.0    // double 泊松比
    // Mullins效应 同 MooneyRivlin
}
```


// - HyperFoam3, 隐式移植的算法, 与HyperFoam2相同，但定义的是Mu^h, 底层算法与Abauqs一致


// - Yeoh, 隐式移植的算法

```json
"MaterialType": "Yeoh",
"MaterialConstants": {
    // 直接定义的方式
    "C10": 0.0,     // double
    "C20": 0.0,
    "C30": 0.0,
    "D1": 0.01,
    "D2": 0.0,
    "D3": 0.0
    // 实验曲线的方式 同 MooneyRivlin
    // Mullins效应 同 MooneyRivlin
}
```


// 热参数，可与材料参数组合

```json
"ThConstants": {
"InitialT": …, // double,初始温度
"SpecificHeat":…,// double,比热容
"SolidThConductivity":…,double[2]，固体导热系数
"HeatTranFormular":…, int，热传导算法
"MeltingT":…, double,融化温度
"LiquidThConductivity":…,double[2]，液体导热系数
"EnerTranFrac":..,double,应变能转化热能比例
"ExpansionFunc":…, string,膨胀系数函数
"ExpansionValue":…, double,膨胀系数值
}
```


//MaterialType == PlasticBrittle【LAW27】

```json
"ElasticModulus": ... ,                     // double, 弹性模量  (0, e38)【E】
"PoissonRatio": ... ,                        // double, 泊松比   (-1, 0.5]【ν】
"YieldStress": ... ,                           // double, 屈服常数  （-e38, +e38）A【a】
"HardeningCoefB": ... ,                  // double, 硬化系数   （-e38, +e38）B【b】
"HardeningExpN": ... ,                   // double, 硬化幂指数  （-e38, +e38）n【n】
"MaxPlasStress":...,                     // double, 最大塑性应力，默认e30
"RateCoef": ... ,                              // double, 应变率相关系数 （-e38, +e38）,默认0【c】
"RefStrainRate": ... ,                      // double, 参考应变率   [0, +e38）,默认1.0【ε0_dot】
"StrainRateFlag":...,                    //bool,应变率是否影响最大应力【ICC】
"FailureStrain": ... ,                  // double[2],两个主应变方向的失效应变 
"MaxFailureStrain": ... ,                  // double[2], 最大失效应变，超过后应力根据损伤计算
"MaxDamageFac": ... ,                  // double[2], 最大损伤因子
"MaxDeleteStrain": ... ,                  // double[2], 删除单元的最大应变
```


