using VRChat.API;
using VRChat.API.Client;
using VRChat.API.Model;
using VRChatOSCLib;

public class VRChatNPC : MonoBehaviour
{
    private ApiClient _apiClient;
    private AuthenticationApi _authApi;
    private UsersApi _userApi;
    private WorldsApi _worldApi;
    private VRChatOSC _osc;

    void Start()
    {
        // Initialize the API client and APIs
        _apiClient = new ApiClient();
        _authApi = new AuthenticationApi(_apiClient, _apiClient, new Configuration());
        _userApi = new UsersApi(_apiClient, _apiClient, new Configuration());
        _worldApi = new WorldsApi(_apiClient, _apiClient, new Configuration());

        // Initialize the OSC client
        _osc = new VRChatOSC();

        // Connect to the OSC server
        _osc.Connect(9000);

        // Authenticate with the VRChat API
        Authenticate();
    }

    void Authenticate()
    {
        // Create a configuration for us to log in
        Configuration config = new Configuration();
        config.Username = "Username";
        config.Password = "Password";
        config.UserAgent = "ExampleProgram/0.0.1 mydiscordusername";

        // Authenticate with the API
        ApiResponse<CurrentUser> currentUserResp = _authApi.GetCurrentUserWithHttpInfo();

        if (requiresEmail2FA(currentUserResp))
        {
            _authApi.Verify2FAEmailCode(new TwoFactorEmailCode("123456"));
        }
        else
        {
            _authApi.Verify2FA(new TwoFactorAuthCode("123456"));
        }
    }

    void Update()
    {
        // Send a message to the OSC server
        _osc.SendTo("/test/lib/float", 0.5f);

        // Send a parameter to the avatar
        _osc.SendParameter("GlassesToggle", true);
    }
}
