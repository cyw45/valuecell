import { Navigate } from "react-router";

export default function RedirectToDashboard() {
  return <Navigate to="/dashboard" replace />;
}
