import { Navigate } from "react-router";

export default function PlatformRedirectPage() {
  return <Navigate replace to="/admin/tenants" />;
}
