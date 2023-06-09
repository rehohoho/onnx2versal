#include "graph_gemm.h"
#include "graph_utils.h"


template <template<int, int, int, int> class GEMM, 
  int M, int K, int N, int IS_RELU>
class GemmReluGraphTest : public adf::graph {

  private:
    GemmReluGraph<GEMM, M, K, N, IS_RELU> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    GemmReluGraphTest(
      const std::string& id,
      std::vector<float> weights,
      std::vector<float> bias,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "gemm_out.txt"
    ): g(weights, bias) { 
      plin[0] = adf::input_plio::create("plin0_gemm"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_gemm"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<M*K*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<M*N*4>> (g.pout[0], plout[0].in[0]);
    }

};


template <template<int, int, int, int> class GEMM, 
  int M, int K, int N, int IS_RELU>
class GemmReluStreamGraphTest : public adf::graph {

  private:
    GemmReluStreamGraph<GEMM, M, K, N, IS_RELU> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    adf::input_gmio gmio_w;

    GemmReluStreamGraphTest(
      const std::string& id,
      std::vector<float> bias,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "gemm_out.txt"
    ): g(bias) { 
      plin[0] = adf::input_plio::create("plin0_gemm"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_gemm"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w = adf::input_gmio::create("gmio0_gemm"+id+"_w", 64, 1000);
      
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmio_w.out[0], g.pin[1]);
      adf::connect<adf::window<M*N*4>> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
const int M = 2;
const int K = 36;
const int N = 10;
const int N_PAD = (N + 3)/4*4;
std::vector<float> fpweights_mknk {0.9767611026763916, 0.6048455238342285, 0.7392635941505432, 0.03918779268860817, 0.28280696272850037, 0.12019655853509903, 0.296140193939209, 0.11872772127389908, 0.3179831802845001, 0.414262980222702, 0.06414749473333359, 0.6924721002578735, 0.5666014552116394, 0.26538950204849243, 0.5232480764389038, 0.09394051134586334, 0.5759465098381042, 0.9292961955070496, 0.3185689449310303, 0.6674103736877441, 0.13179786503314972, 0.7163271903991699, 0.28940609097480774, 0.18319135904312134, 0.5865129232406616, 0.02010754682123661, 0.8289400339126587, 0.004695476032793522, 0.6778165102005005, 0.2700079679489136, 0.7351940274238586, 0.9621885418891907, 0.2487531453371048, 0.5761573314666748, 0.5920419096946716, 0.5722519159317017, 0.22308163344860077, 0.9527490139007568, 0.4471253752708435, 0.8464086651802063, 0.6994792819023132, 0.2974369525909424, 0.8137978315353394, 0.396505743265152, 0.8811032176017761, 0.5812729001045227, 0.8817353844642639, 0.6925315856933594, 0.7252542972564697, 0.5013243556022644, 0.9560836553573608, 0.6439902186393738, 0.4238550364971161, 0.6063932180404663, 0.019193198531866074, 0.30157482624053955, 0.6601735353469849, 0.2900775969028473, 0.6180154085159302, 0.42876869440078735, 0.1354740709066391, 0.29828232526779175, 0.5699648857116699, 0.5908727645874023, 0.5743252635002136, 0.6532008051872253, 0.6521032452583313, 0.43141844868659973, 0.8965466022491455, 0.36756187677383423, 0.4358649253845215, 0.8919233679771423, 0.806194007396698, 0.7038885951042175, 0.10022688657045364, 0.9194825887680054, 0.7142413258552551, 0.9988470077514648, 0.14944830536842346, 0.8681260347366333, 0.16249293088912964, 0.6155595779418945, 0.1238199844956398, 0.8480082154273987, 0.8073189854621887, 0.5691007375717163, 0.40718328952789307, 0.06916699558496475, 0.6974287629127502, 0.45354267954826355, 0.7220556139945984, 0.8663823008537292, 0.9755215048789978, 0.855803370475769, 0.011714084073901176, 0.359978049993515, 0.729990541934967, 0.17162968218326569, 0.5210366249084473, 0.054337989538908005, 0.19999653100967407, 0.01852179504930973, 0.793697714805603, 0.2239246815443039, 0.3453516662120819, 0.9280812740325928, 0.704414427280426, 0.031838931143283844, 0.1646941602230072, 0.6214783787727356, 0.5772286057472229, 0.23789282143115997, 0.9342139959335327, 0.6139659285545349, 0.5356327891349792, 0.5899099707603455, 0.7301220297813416, 0.31194499135017395, 0.39822107553482056, 0.20984375476837158, 0.18619300425052643, 0.9443724155426025, 0.739550769329071, 0.49045881628990173, 0.22741462290287018, 0.2543564736843109, 0.058029159903526306, 0.43441662192344666, 0.3117958903312683, 0.6963434815406799, 0.37775182723999023, 0.1796036809682846, 0.024678727611899376, 0.06724963337182999, 0.6793927550315857, 0.4536968469619751, 0.5365791916847229, 0.8966712951660156, 0.990338921546936, 0.21689698100090027, 0.6630781888961792, 0.2633223831653595, 0.02065099962055683, 0.7583786249160767, 0.32001715898513794, 0.38346388936042786, 0.5883170962333679, 0.8310484290122986, 0.6289818286895752, 0.872650682926178, 0.27354204654693604, 0.7980468273162842, 0.18563593924045563, 0.9527916312217712, 0.6874882578849792, 0.21550767123699188, 0.9473705887794495, 0.7308558225631714, 0.2539416551589966, 0.21331197023391724, 0.518200695514679, 0.02566271834075451, 0.20747007429599762, 0.4246854782104492, 0.3741699755191803, 0.46357542276382446, 0.27762871980667114, 0.5867843627929688, 0.8638556003570557, 0.11753185838460922, 0.517379105091095, 0.13206811249256134, 0.7168596982955933, 0.39605969190597534, 0.5654212832450867, 0.1832798421382904, 0.14484776556491852, 0.4880562722682953, 0.35561272501945496, 0.9404319524765015, 0.7653252482414246, 0.748663604259491, 0.9037197232246399, 0.08342243731021881, 0.5521924495697021, 0.5844760537147522, 0.961936354637146, 0.29214751720428467, 0.24082878232002258, 0.10029394179582596, 0.016429629176855087, 0.9295293092727661, 0.669916570186615, 0.7851529121398926, 0.28173011541366577, 0.5864101648330688, 0.06395526975393295, 0.48562759160995483, 0.9774951338768005, 0.8765052556991577, 0.3381589651107788, 0.961570143699646, 0.23170162737369537, 0.9493188261985779, 0.9413776993751526, 0.799202561378479, 0.6304479241371155, 0.8742879629135132, 0.2930202782154083, 0.8489435315132141, 0.6178767085075378, 0.013236857950687408, 0.34723350405693054, 0.14814086258411407, 0.9818294048309326, 0.4783703088760376, 0.49739137291908264, 0.6394725441932678, 0.36858460307121277, 0.13690027594566345, 0.8221177458763123, 0.1898479163646698, 0.5113189816474915, 0.2243170291185379, 0.09784448146820068, 0.8621914982795715, 0.9729194641113281, 0.9608346819877625, 0.9065554738044739, 0.774047315120697, 0.3331451416015625, 0.08110138773918152, 0.40724116563796997, 0.2322341352701187, 0.13248763978481293, 0.053427182137966156, 0.7255943417549133, 0.011427458375692368, 0.7705807685852051, 0.14694663882255554, 0.07952208071947098, 0.08960303664207458, 0.6720477938652039, 0.24536721408367157, 0.4205394685268402, 0.557368814945221, 0.8605511784553528, 0.7270442843437195, 0.2703278958797455, 0.131482794880867, 0.05537432059645653, 0.3015986382961273, 0.2621181607246399, 0.45614057779312134, 0.6832813620567322, 0.6956254243850708, 0.28351885080337524, 0.3799269497394562, 0.18115095794200897, 0.7885454893112183, 0.05684807524085045, 0.6969972252845764, 0.7786954045295715, 0.7774075865745544, 0.25942257046699524, 0.3738131523132324, 0.5875996351242065, 0.27282190322875977, 0.3708527982234955, 0.19705428183078766, 0.4598558843135834, 0.044612299650907516, 0.7997958660125732, 0.07695644348859787, 0.5188351273536682, 0.3068101108074188, 0.5775429606437683, 0.9594333171844482, 0.6455702185630798, 0.03536243736743927, 0.4304024279117584, 0.5100168585777283, 0.5361775159835815, 0.6813924908638, 0.2775960862636566, 0.12886056303977966, 0.3926756680011749, 0.9564056992530823, 0.1871308982372284, 0.9039839506149292, 0.5438059568405151, 0.4569114148616791, 0.8820413947105408, 0.45860394835472107, 0.7241676449775696, 0.3990253210067749, 0.9040443897247314, 0.6900250315666199, 0.6996220350265503, 0.32772040367126465, 0.7567786574363708, 0.6360610723495483, 0.2400202751159668, 0.16053882241249084, 0.796391487121582, 0.9591665863990784, 0.4581388235092163, 0.5909841656684875, 0.8577226400375366, 0.45722344517707825, 0.9518744945526123, 0.5757511854171753, 0.8207671046257019, 0.9088436961174011, 0.8155238032341003, 0.15941447019577026, 0.6288984417915344, 0.39843425154685974, 0.06271295249462128, 0.4240322411060333, 0.25868406891822815, 0.849038302898407, 0.03330462798476219, 0.9589827060699463, 0.35536885261535645, 0.3567068874835968, 0.01632850244641304, 0.18523232638835907, 0.40125951170921326, 0.9292914271354675, 0.0996149331331253, 0.9453015327453613, 0.869488537311554, 0.4541623890399933, 0.326700896024704, 0.23274412751197815, 0.6144647002220154, 0.03307459130883217, 0.015606064349412918, 0.428795725107193, 0.06807407736778259, 0.2519409954547882, 0.2211609184741974, 0.253191202878952, 0.13105523586273193, 0.012036222964525223, 0.11548429727554321, 0.6184802651405334, 0.9742562174797058, 0.9903450012207031, 0.40905410051345825, 0.1629544198513031, 0.6387617588043213, 0.4903053343296051, 0.9894098043441772, 0.06530420482158661, 0.7832344174385071, 0.28839850425720215, 0.24141861498355865, 0.6625045537948608};
std::vector<float> fpweights_mkkn_pad {0.24606318771839142, 0.6658591032028198, 0.5173085331916809, 0.4240889847278595, 0.5546877980232239, 0.2870515286922455, 0.7065746784210205, 0.414856880903244, 0.3605455458164215, 0.8286569118499756, 0.0, 0.0, 0.9249669313430786, 0.04600730910897255, 0.2326269894838333, 0.34851935505867004, 0.8149664998054504, 0.9854914546012878, 0.9689717292785645, 0.904948353767395, 0.2965562641620636, 0.9920112490653992, 0.0, 0.0, 0.24942004680633545, 0.10590615123510361, 0.9509525895118713, 0.2334202527999878, 0.6897682547569275, 0.05835635960102081, 0.7307090759277344, 0.8817201852798462, 0.27243688702583313, 0.3790569007396698, 0.0, 0.0, 0.3742961883544922, 0.7487882375717163, 0.2378072440624237, 0.17185309529304504, 0.4492916464805603, 0.30446839332580566, 0.8391891121864319, 0.23774182796478271, 0.5023894309997559, 0.9425836205482483, 0.0, 0.0, 0.6339976787567139, 0.8672894239425659, 0.940209686756134, 0.7507648468017578, 0.6995750665664673, 0.9679655432701111, 0.9944007992744446, 0.4518216848373413, 0.07086978107690811, 0.29279401898384094, 0.0, 0.0, 0.15235470235347748, 0.41748636960983276, 0.13128933310508728, 0.6041178107261658, 0.38280805945396423, 0.8953858613967896, 0.96779465675354, 0.5468848943710327, 0.2748235762119293, 0.5922304391860962, 0.0, 0.0, 0.8967611789703369, 0.40673333406448364, 0.5520782470703125, 0.2716527581214905, 0.4554441571235657, 0.4017135500907898, 0.24841345846652985, 0.5058664083480835, 0.31038081645965576, 0.37303486466407776, 0.0, 0.0, 0.5249704718589783, 0.7505950331687927, 0.3335074782371521, 0.9241587519645691, 0.8623185753822327, 0.048690296709537506, 0.2536425292491913, 0.4461355209350586, 0.10462789237499237, 0.34847599267959595, 0.0, 0.0, 0.7400975227355957, 0.6805144548416138, 0.6223844289779663, 0.7105283737182617, 0.20492368936538696, 0.3416981101036072, 0.676242470741272, 0.879234790802002, 0.5436780452728271, 0.2826996445655823, 0.0, 0.0, 0.030235258862376213, 0.7103368043899536, 0.007884103804826736, 0.37267908453941345, 0.5305371880531311, 0.922111451625824, 0.08949454873800278, 0.40594232082366943, 0.024313200265169144, 0.3426109850406647, 0.0, 0.0, 0.6222310662269592, 0.2790679335594177, 0.2097499519586563, 0.11570323258638382, 0.5771402716636658, 0.6952700018882751, 0.6719571352005005, 0.9488610029220581, 0.002703213831409812, 0.6471966505050659, 0.0, 0.0, 0.60039222240448, 0.5887396335601807, 0.9627703428268433, 0.016871673986315727, 0.6964824199676514, 0.8136786222457886, 0.5098071694374084, 0.33396488428115845, 0.7908401489257812, 0.09724292904138565, 0.0, 0.0, 0.44203564524650574, 0.5199523568153381, 0.6939564347267151, 0.09088572859764099, 0.2277594953775406, 0.4103015661239624, 0.6232946515083313, 0.8869608044624329, 0.618826150894165, 0.13346147537231445, 0.0, 0.0, 0.9805801510810852, 0.8717857599258423, 0.5027207732200623, 0.9223479628562927, 0.5413808226585388, 0.9233060479164124, 0.8298973441123962, 0.968286395072937, 0.919782817363739, 0.03603381663560867, 0.0, 0.0, 0.1747720092535019, 0.3891346752643585, 0.9521427154541016, 0.300028920173645, 0.16046763956546783, 0.8863046765327454, 0.4463944137096405, 0.9078755974769592, 0.16023047268390656, 0.6611174941062927, 0.0, 0.0, 0.4402637481689453, 0.07648676633834839, 0.6964631676673889, 0.2473987489938736, 0.03961552307009697, 0.05994429811835289, 0.06107853725552559, 0.9077329635620117, 0.7398838996887207, 0.8980623483657837, 0.0, 0.0, 0.6725823283195496, 0.5289399027824402, 0.30444636940956116, 0.997962236404419, 0.36218905448913574, 0.47064894437789917, 0.37824517488479614, 0.979526937007904, 0.1746583878993988, 0.32798799872398376, 0.0, 0.0, 0.6803486943244934, 0.06320761889219284, 0.60724937915802, 0.47764649987220764, 0.2839999794960022, 0.2384132742881775, 0.5145127177238464, 0.36792758107185364, 0.4565199017524719, 0.3374773859977722, 0.0, 0.0, 0.9704936742782593, 0.13343943655490875, 0.09680395573377609, 0.3433917164802551, 0.5910269021987915, 0.6591764688491821, 0.3972567617893219, 0.9992780089378357, 0.35189300775527954, 0.7214066386222839, 0.0, 0.0, 0.6375827193260193, 0.8130538463592529, 0.9762256741523743, 0.8897936344146729, 0.7645619511604309, 0.6982485055923462, 0.335498183965683, 0.14768557250499725, 0.06263600289821625, 0.2419017106294632, 0.0, 0.0, 0.432281494140625, 0.521996259689331, 0.7730835676193237, 0.9587409496307373, 0.1173204779624939, 0.10700414329767227, 0.5896947383880615, 0.7453980445861816, 0.848150372505188, 0.9358320832252502, 0.0, 0.0, 0.9834262132644653, 0.39980170130729675, 0.3803351819515228, 0.14780867099761963, 0.6849344372749329, 0.6567619442939758, 0.8620625734329224, 0.09725799411535263, 0.49777689576148987, 0.5810819268226624, 0.0, 0.0, 0.2415570467710495, 0.16902540624141693, 0.8595808148384094, 0.05853492394089699, 0.47062090039253235, 0.11583399772644043, 0.45705875754356384, 0.9799623489379883, 0.4237063527107239, 0.857124924659729, 0.0, 0.0, 0.11731556057929993, 0.2712520658969879, 0.40379273891448975, 0.39981213212013245, 0.6713835000991821, 0.3447181284427643, 0.713766872882843, 0.6391869187355042, 0.399161159992218, 0.43176013231277466, 0.0, 0.0, 0.614527702331543, 0.0700421929359436, 0.8224067091941833, 0.65342116355896, 0.7263424396514893, 0.5369229912757874, 0.11047711223363876, 0.4050356149673462, 0.40537357330322266, 0.3210429847240448, 0.0, 0.0, 0.029950324445962906, 0.73725426197052, 0.10978446155786514, 0.6063081622123718, 0.7032175064086914, 0.6347863078117371, 0.95914226770401, 0.10329815745353699, 0.8671671748161316, 0.02919023483991623, 0.0, 0.0, 0.534916877746582, 0.4042436182498932, 0.5241838693618774, 0.36509987711906433, 0.19056691229343414, 0.01912289671599865, 0.5181497931480408, 0.8427768349647522, 0.3732159435749054, 0.2228638231754303, 0.0, 0.0, 0.080532006919384, 0.0853109210729599, 0.22139644622802734, 0.10001406073570251, 0.26503971219062805, 0.06614946573972702, 0.06560486555099487, 0.8562761545181274, 0.1621202677488327, 0.5596824288368225, 0.0, 0.0, 0.7734555602073669, 0.4564095735549927, 0.15336887538433075, 0.19959613680839539, 0.43298420310020447, 0.52823406457901, 0.3494403064250946, 0.7814795970916748, 0.7510216236114502, 0.9272118210792542, 0.0, 0.0, 0.028952548280358315, 0.8956912755966187, 0.39256879687309265, 0.8783724904060364, 0.690784752368927, 0.987348735332489, 0.7592824697494507, 0.3645446300506592, 0.5010631680488586, 0.37638914585113525, 0.0, 0.0, 0.364911824464798, 0.2609044909477234, 0.49597030878067017, 0.6817399263381958, 0.27734026312828064, 0.5243797898292542, 0.117380291223526, 0.1598452925682068, 0.04680635407567024, 0.9707314372062683, 0.0, 0.0, 0.0038603513967245817, 0.17857997119426727, 0.6128667593002319, 0.08136960119009018, 0.8818964958190918, 0.7196201682090759, 0.9663899540901184, 0.5076355338096619, 0.3004036843776703, 0.549500584602356, 0.0, 0.0, 0.9308187365531921, 0.5207614302635193, 0.2672070264816284, 0.8773987889289856, 0.3719187378883362, 0.0013833499979227781, 0.2476850152015686, 0.31823351979255676, 0.8587774634361267, 0.4585031569004059, 0.0, 0.0, 0.4445872902870178, 0.33610227704048157, 0.880678117275238, 0.9450267553329468, 0.9918903112411499, 0.3767412602901459, 0.9661474227905273, 0.7918795943260193, 0.675689160823822, 0.24488948285579681, 0.0, 0.0, 0.21645726263523102, 0.1660478264093399, 0.9227566123008728, 0.2940766513347626, 0.4530942440032959, 0.49395784735679626, 0.7781715989112854, 0.8442349433898926, 0.1390727013349533, 0.4269043505191803, 0.0, 0.0, 0.842854917049408, 0.8180332779884338, 0.10241375863552094, 0.15638335049152374, 0.304198682308197, 0.07535906881093979, 0.4246630072593689, 0.10761770606040955, 0.5682175755500793, 0.24655693769454956, 0.0, 0.0};
std::vector<float> fpbias {0.5964330434799194, 0.11752564460039139, 0.9758838415145874, 0.9325612187385559, 0.39179694652557373, 0.24217858910560608, 0.2503982186317444, 0.4833935499191284, 0.0399928018450737, 0.6397051215171814};

// MK * NK
GemmReluGraphTest<GemmReluScalarMKNK, M, K, N, 1> gemmReluScalarMKNK(
  "gemmReluScalarMKNK", fpweights_mknk, fpbias, 
  "gemm_fpin.txt", "gemmMKNK_fpout_shape2x10_GemmReluScalarMKNK.txt");

GemmReluStreamGraphTest<GemmReluScalarMKNKStream, M, K, N, 1> gemmReluScalarGmemParamMKNK(
  "gemmReluScalarGmemParamMKNK", fpbias,
  "gemm_fpin.txt", "gemmMKNK_fpout_shape2x10_GemmReluScalarMKNKStream.txt");

// MK * KN
GemmReluGraphTest<GemmReluScalarMKKN, M, K, N_PAD, 1> gemmReluScalarMKKN(
  "gemmReluScalarMKKN", fpweights_mkkn_pad, fpbias, 
  "gemm_fpin.txt", "gemmMKKN_fpout_shape2x10_GemmReluScalarMKKN.txt");

GemmReluGraphTest<GemmReluMKKN, M, K, N_PAD, 1> gemmReluMKKN(
  "gemmReluMKKN", fpweights_mkkn_pad, fpbias, 
  "gemm_fpin.txt", "gemmMKKN_fpout_shape2x10_GemmReluMKKN.txt");



#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // init gmio
  float_t* mknk_buf = (float_t *) adf::GMIO::malloc(M*K*N*sizeof(float_t));
  for (int i = 0; i < M; i++) {
    memcpy(mknk_buf + i*K*N, fpweights_mknk.data(), K*N*sizeof(float_t));
  }

  // MK * NK
  adfCheck(gemmReluScalarMKNK.init(), "init gemmReluScalarMKNK");
  adfCheck(gemmReluScalarMKNK.run(ITER_CNT), "run gemmReluScalarMKNK");
	adfCheck(gemmReluScalarMKNK.end(), "end gemmReluScalarMKNK");

  adfCheck(gemmReluScalarGmemParamMKNK.init(), "init gemmReluScalarGmemParamMKNK");
  gemmReluScalarGmemParamMKNK.gmio_w.gm2aie_nb(mknk_buf, M*K*N*sizeof(float_t));
  adfCheck(gemmReluScalarGmemParamMKNK.run(ITER_CNT), "run gemmReluScalarGmemParamMKNK");
	adfCheck(gemmReluScalarGmemParamMKNK.end(), "end gemmReluScalarGmemParamMKNK");

  // MK * KN
  adfCheck(gemmReluScalarMKKN.init(), "init gemmReluScalarMKKN");
  adfCheck(gemmReluScalarMKKN.run(ITER_CNT), "run gemmReluScalarMKKN");
	adfCheck(gemmReluScalarMKKN.end(), "end gemmReluScalarMKKN");

  adfCheck(gemmReluMKKN.init(), "init gemmReluMKKN");
  adfCheck(gemmReluMKKN.run(ITER_CNT), "run gemmReluMKKN");
	adfCheck(gemmReluMKKN.end(), "end gemmReluMKKN");
  
  // cleanup gmio
  adf::GMIO::free(mknk_buf);
  return 0;
}
#endif
