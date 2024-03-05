import useHttp from './useHttp';
import {
  CreateBotPublicationApiKeyResponse,
  DeleteBotPublicationApiKeyResponse,
  DeleteBotPublicationRequest,
  DeleteBotPublicationResponse,
  GetBotPublicationApiKeyResponse,
  GetBotPublicationRequest,
  GetBotPublicationRespose,
  ListPublicBotsRequest,
  ListPublicBotsResponse,
  PublishBotRequest,
  PublishBotResponse,
} from '../@types/api-publication';

const useBotPublicationApi = () => {
  const http = useHttp();

  return {
    listPublicBots: (req: ListPublicBotsRequest) => {
      return http.get<ListPublicBotsResponse>(['/admin/public-bots', req]);
    },
    getBotPublication: (botId?: string, req?: GetBotPublicationRequest) => {
      return http.get<GetBotPublicationRespose>(
        botId ? [`/bot/${botId}/publication`, req] : null
      );
    },
    publishBot: (botId: string, req: PublishBotRequest) => {
      return http.post<PublishBotResponse>(`/bot/${botId}/publication`, req);
    },
    deleteBotPublication: (
      botId: string,
      req?: DeleteBotPublicationRequest
    ) => {
      return http.delete<DeleteBotPublicationResponse>(
        `/bot/${botId}/publication`,
        req
      );
    },
    getBotPublicationApiKey: (botId?: string, apiKeyId?: string) => {
      return http.get<GetBotPublicationApiKeyResponse>(
        botId && apiKeyId
          ? `/bot/${botId}/publication/api-key/${apiKeyId}`
          : null
      );
    },
    deleteBotPublicationApiKey: (botId: string, apiKeyId: string) => {
      return http.delete<DeleteBotPublicationApiKeyResponse>(
        `/bot/${botId}/publication/api-key/${apiKeyId}`
      );
    },
    createBotPublicationApiKey: (botId: string) => {
      return http.post<CreateBotPublicationApiKeyResponse>(
        `/bot/${botId}/publication/api-key`,
        {}
      );
    },
  };
};

export default useBotPublicationApi;