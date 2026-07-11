export interface SaaSRegisterRequest {
  email: string;
  password: string;
  workspace_name: string;
}

export interface SaaSLoginRequest {
  email: string;
  password: string;
}

export interface SaaSAuthResponse {
  access_token: string;
  user_id: string;
  tenant_id: string;
  email: string;
}

export interface SaaSMeResponse {
  user_id: string;
  tenant_id: string;
  email: string;
}
