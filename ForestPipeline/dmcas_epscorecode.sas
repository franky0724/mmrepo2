/*----------------------------------------------------------------------------------*/
/* Product:            Visual Data Mining and Machine Learning                      */
/* Release Version:    V2020.1.2                                                    */
/* Component Version:  V2020.1.1                                                    */
/* CAS Version:        V.04.00M0P01052021                                           */
/* SAS Version:        V.04.00M0P010521                                             */
/* Site Number:        70180938                                                     */
/* Host:               controller.sas-cas-server-default.base.svc.cluster.local     */
/* Encoding:           utf-8                                                        */
/* Java Encoding:      UTF8                                                         */
/* Locale:             en_US                                                        */
/* Project GUID:       5d79a6e8-28ae-405a-871b-ed57cde2bffa                         */
/* Node GUID:          9d1ef204-a525-4e69-bee4-5822551f50f0                         */
/* Node Id:            9AVB56RG138BYEX3DURBJXWC0                                    */
/* Algorithm:          Forest                                                       */
/* Generated by:       scnkuj                                                       */
/* Date:               13JAN2021:06:49:40                                           */
/*----------------------------------------------------------------------------------*/
data sasep.out;
   dcl package score _9AVB56RG138BYEX3DURBJXWC0();
   dcl double "P_BAD1" having label n'Predicted: BAD=1';
   dcl double "P_BAD0" having label n'Predicted: BAD=0';
   dcl nchar(32) "I_BAD" having label n'Into: BAD';
   dcl nchar(4) "_WARN_" having label n'Warnings';
   dcl double EM_EVENTPROBABILITY;
   dcl nchar(12) EM_CLASSIFICATION;
   dcl double EM_PROBABILITY;
   varlist allvars [_all_];
 
    
   method init();
      _9AVB56RG138BYEX3DURBJXWC0.setvars(allvars);
      _9AVB56RG138BYEX3DURBJXWC0.setkey(n'D44CA451EA46EB9076E1DD40F8C9ED098DE3ABE5');
   end;
    
   method post_9AVB56RG138BYEX3DURBJXWC0();
      dcl double _P_;
       
      if "P_BAD0" = . then "P_BAD0" = 0.7890939597;
      if "P_BAD1" = . then "P_BAD1" = 0.2109060403;
      if MISSING("I_BAD") then do ;
      _P_ = 0.0;
      if "P_BAD1" > _P_ then do ;
      _P_ = "P_BAD1";
      "I_BAD" = '           1';
      end;
      if "P_BAD0" > _P_ then do ;
      _P_ = "P_BAD0";
      "I_BAD" = '           0';
      end;
      end;
      EM_EVENTPROBABILITY = "P_BAD1";
      EM_CLASSIFICATION = "I_BAD";
      EM_PROBABILITY = MAX("P_BAD1", "P_BAD0");
    
   end;
    
 
   method run();
      set SASEP.IN;
      _9AVB56RG138BYEX3DURBJXWC0.scoreRecord();
      post_9AVB56RG138BYEX3DURBJXWC0();
   end;
 
   method term();
   end;
 
enddata;