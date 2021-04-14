#include "connector.h"
#include "forward_declare.h"
#include <vector>

namespace snnl {

template <class TElem>
class TModel {

    std::vector<TConnectorShPtr<TElem>> _connectors;

public:
    template <template <class> class TChildConnector, typename... TArgs>
    ::std::shared_ptr<TChildConnector<TElem>> registerConnector(TArgs&&... args)
    {
        std::shared_ptr<TChildConnector<TElem>> conn =
            TConnector<TElem>::template create<TChildConnector>(
                std::forward<TArgs>(args)...);
        _connectors.push_back(conn);
        return conn;
    }

    virtual TNodeShPtr<TElem> call(std::vector<TNodeShPtr<TElem>> inputs) = 0;
};

} // namespace snnl