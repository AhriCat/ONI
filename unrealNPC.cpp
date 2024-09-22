#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "GameFramework/PlayerController.h"
#include "ChatbotSystem.generated.h"

UCLASS()
class CHATBOTSYSTEM_API AChatbotSystem : public AActor
{
    GENERATED_BODY()

public:
    AChatbotSystem();

protected:
    virtual void BeginPlay() override;

public:
    virtual void Tick(float DeltaTime) override;

    UFUNCTION(BlueprintCallable, Category = "Chatbot")
    void SendMessage(FString Message);

    UFUNCTION(BlueprintCallable, Category = "Chatbot")
    FString GetResponse(FString Message);

private:
    FString ProcessMessage(FString Message);

    FString GenerateResponse(FString Message);

    UserManager* UserManager;
};

AChatbotSystem::AChatbotSystem()
{
    PrimaryActorTick.bCanEverTick = true;
}

void AChatbotSystem::BeginPlay()
{
    Super::BeginPlay();

    UserManager = NewObject<UserManager>(this, FName("UserManager"));
}

void AChatbotSystem::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // Process any incoming messages
    while ( UserManager->HasNewMessages() )
    {
        FString Message = UserManager->GetNextMessage();
        ProcessMessage(Message);
    }
}

void AChatbotSystem::SendMessage(FString Message)
{
    // Send the message to the user manager
    UserManager->SendMessage(Message);
}

FString AChatbotSystem::GetResponse(FString Message)
{
    return GenerateResponse(Message);
}

FString AChatbotSystem::ProcessMessage(FString Message)
{
    // Process the message and generate a response
    return GenerateResponse(Message);
}

FString AChatbotSystem::GenerateResponse(FString Message)
{
    // This is where you would implement your chatbot's logic to generate a response
    // For now, just return a default response
    return "Hello!";
}

UCLASS()
class CHATBOTSYSTEM_API UserManager : public UObject
{
    GENERATED_BODY()

public:
    UserManager(AChatbotSystem* ChatbotSystem);

    void SendMessage(FString Message);

    bool HasNewMessages();

    FString GetNextMessage();

private:
    AChatbotSystem* ChatbotSystem;
    TArray<FString> MessageQueue;
};

UserManager::UserManager(AChatbotSystem* ChatbotSystem)
{
    this->ChatbotSystem = ChatbotSystem;
}

void UserManager::SendMessage(FString Message)
{
    MessageQueue.Add(Message);
}

bool UserManager::HasNewMessages()
{
    return MessageQueue.Num() > 0;
}

FString UserManager::GetNextMessage()
{
    if (HasNewMessages())
    {
        FString Message = MessageQueue[0];
        MessageQueue.RemoveAt(0);
        return Message;
    }
    return "";
}