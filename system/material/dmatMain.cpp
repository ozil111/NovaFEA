#include "dmatMain.h"
#include "components/material_components.h"
#include <spdlog/spdlog.h>
#include "material/mat1/LinearElasticMatrixSystem.h"

void dmat_main(entt::registry &registry, entt::entity material_entity) {
  if (!registry.all_of<Component::MaterialModel>(material_entity)) {
    spdlog::warn(
        "dmat_main: material entity has no MaterialModel component, skipping.");
    return;
  }
  const auto &model = registry.get<Component::MaterialModel>(material_entity);
  const int typeid_val = Component::material_typeid_from_model(model.value);

  switch (typeid_val) {
  case 1:
    compute_single_linear_elastic_matrix(registry, material_entity);
    break;
  default:
    spdlog::warn("dmat_main: unknown material type '{}' (typeid={}), skipping.",
                 model.value, typeid_val);
    break;
  }
}